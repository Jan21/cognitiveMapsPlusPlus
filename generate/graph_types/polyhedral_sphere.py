import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate(sphere_type="icosphere", subdivisions=2, num_latitude=20, num_longitude=20):
    """
    Generate a polyhedral sphere graph.

    Supports multiple sphere generation methods:
    - icosphere: Subdivided icosahedron (most uniform distribution)
    - uv_sphere: UV sphere with latitude/longitude lines
    - cube_sphere: Subdivided cube projected onto sphere
    - octahedron: Subdivided octahedron

    Args:
        sphere_type: Type of sphere ("icosphere", "uv_sphere", "cube_sphere", "octahedron")
        subdivisions: Number of subdivisions (for icosphere, cube_sphere, octahedron)
        num_latitude: Number of latitude circles (for uv_sphere)
        num_longitude: Number of longitude circles (for uv_sphere)

    Returns:
        NetworkX graph representing the wireframe sphere
    """
    if sphere_type == "icosphere":
        return _generate_icosphere(subdivisions)
    elif sphere_type == "uv_sphere":
        return _generate_uv_sphere(num_latitude, num_longitude)
    elif sphere_type == "cube_sphere":
        return _generate_cube_sphere(subdivisions)
    elif sphere_type == "octahedron":
        return _generate_octahedron_sphere(subdivisions)
    else:
        raise ValueError(f"Unknown sphere type: {sphere_type}. "
                        f"Supported: icosphere, uv_sphere, cube_sphere, octahedron")


def _generate_icosphere(subdivisions=2):
    """
    Generate an icosphere (geodesic sphere) by subdividing an icosahedron.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Initial icosahedron vertices (12 vertices)
    vertices = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])

    # Initial icosahedron faces (20 triangular faces)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    # Subdivide faces
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        vertex_list = list(vertices)

        def get_midpoint(v1_idx, v2_idx):
            """Get or create midpoint between two vertices"""
            edge = tuple(sorted([v1_idx, v2_idx]))

            if edge not in edge_midpoints:
                # Create new vertex at midpoint
                v1 = vertex_list[v1_idx]
                v2 = vertex_list[v2_idx]
                midpoint = (v1 + v2) / 2

                # Project onto unit sphere
                midpoint = midpoint / np.linalg.norm(midpoint)

                # Add to vertex list
                edge_midpoints[edge] = len(vertex_list)
                vertex_list.append(midpoint)

            return edge_midpoints[edge]

        # Subdivide each face into 4 triangles
        for face in faces:
            v0, v1, v2 = face

            # Get midpoints of edges
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)

            # Create 4 new triangular faces
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])

        faces = new_faces
        vertices = np.array(vertex_list)

    # Build graph from faces
    G = nx.Graph()

    # Add vertices as nodes
    for i, pos in enumerate(vertices):
        G.add_node(i, pos=tuple(pos))

    # Add edges from faces
    edges_set = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges_set.add(edge)

    G.add_edges_from(edges_set)

    return G


def _generate_uv_sphere(num_latitude=20, num_longitude=20):
    """
    Generate a UV sphere wireframe graph with latitude and longitude lines.
    """
    G = nx.Graph()

    vertices = []
    node_id = 0

    # Add north pole
    north_pole = node_id
    vertices.append((0, 0, 1))
    G.add_node(node_id, pos=(0, 0, 1))
    node_id += 1

    # Add vertices for latitude circles (excluding poles)
    latitude_rings = []
    for i in range(1, num_latitude):
        # Latitude angle from north pole
        theta = np.pi * i / num_latitude
        z = np.cos(theta)
        r = np.sin(theta)  # radius at this height

        ring_nodes = []
        for j in range(num_longitude):
            phi = 2 * np.pi * j / num_longitude
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            vertices.append((x, y, z))
            G.add_node(node_id, pos=(x, y, z))
            ring_nodes.append(node_id)
            node_id += 1

        latitude_rings.append(ring_nodes)

    # Add south pole
    south_pole = node_id
    vertices.append((0, 0, -1))
    G.add_node(node_id, pos=(0, 0, -1))
    node_id += 1

    # Add horizontal edges (latitude circles)
    for ring in latitude_rings:
        for j in range(num_longitude):
            current = ring[j]
            next_node = ring[(j + 1) % num_longitude]
            G.add_edge(current, next_node)

    # Add vertical edges (longitude lines)
    for j in range(num_longitude):
        # Connect north pole to first ring
        if latitude_rings:
            G.add_edge(north_pole, latitude_rings[0][j])

        # Connect between latitude rings
        for i in range(len(latitude_rings) - 1):
            G.add_edge(latitude_rings[i][j], latitude_rings[i + 1][j])

        # Connect last ring to south pole
        if latitude_rings:
            G.add_edge(latitude_rings[-1][j], south_pole)

    return G


def _generate_cube_sphere(subdivisions=2):
    """
    Generate a cube-sphere by subdividing a cube and projecting onto a sphere.
    """
    G = nx.Graph()

    vertices = []
    vertex_map = {}

    def add_vertex(pos):
        """Add vertex and return its index"""
        # Project onto unit sphere
        pos = np.array(pos)
        pos = pos / np.linalg.norm(pos)
        pos_tuple = tuple(pos)

        # Check if vertex already exists (with tolerance)
        for existing_pos, idx in vertex_map.items():
            if np.linalg.norm(np.array(existing_pos) - pos) < 1e-6:
                return idx

        # Add new vertex
        idx = len(vertices)
        vertices.append(pos)
        vertex_map[pos_tuple] = idx
        G.add_node(idx, pos=pos_tuple)
        return idx

    # Generate vertices on each face of the cube
    n = subdivisions

    # Define the 6 cube faces with their orientations
    faces_data = [
        ([1, 0, 0], [0, 1, 0], 1, [0, 0, 1]),   # +Z face
        ([1, 0, 0], [0, 1, 0], -1, [0, 0, 1]),  # -Z face
        ([0, 0, 1], [0, 1, 0], 1, [1, 0, 0]),   # +X face
        ([0, 0, 1], [0, 1, 0], -1, [1, 0, 0]),  # -X face
        ([1, 0, 0], [0, 0, 1], 1, [0, 1, 0]),   # +Y face
        ([1, 0, 0], [0, 0, 1], -1, [0, 1, 0]),  # -Y face
    ]

    face_grids = []

    for u_axis, v_axis, w_val, w_axis in faces_data:
        u_axis = np.array(u_axis)
        v_axis = np.array(v_axis)
        w_axis = np.array(w_axis)

        grid = []
        for i in range(n + 1):
            row = []
            for j in range(n + 1):
                # Coordinates on the cube face (from -1 to 1)
                u = -1 + 2 * i / n
                v = -1 + 2 * j / n

                # 3D position on cube face
                pos = u * u_axis + v * v_axis + w_val * w_axis

                # Add vertex (will be projected to sphere)
                idx = add_vertex(pos)
                row.append(idx)
            grid.append(row)
        face_grids.append(grid)

    # Add edges within each face
    for grid in face_grids:
        for i in range(n + 1):
            for j in range(n + 1):
                # Horizontal edge
                if j < n:
                    G.add_edge(grid[i][j], grid[i][j + 1])
                # Vertical edge
                if i < n:
                    G.add_edge(grid[i][j], grid[i + 1][j])

    return G


def _generate_octahedron_sphere(subdivisions=2):
    """
    Generate a sphere by subdividing an octahedron.
    """
    # Initial octahedron vertices (6 vertices)
    vertices = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)

    # Initial octahedron faces (8 triangular faces)
    faces = [
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5]
    ]

    # Subdivide using same approach as icosphere
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        vertex_list = list(vertices)

        def get_midpoint(v1_idx, v2_idx):
            """Get or create midpoint between two vertices"""
            edge = tuple(sorted([v1_idx, v2_idx]))

            if edge not in edge_midpoints:
                v1 = vertex_list[v1_idx]
                v2 = vertex_list[v2_idx]
                midpoint = (v1 + v2) / 2

                # Project onto unit sphere
                midpoint = midpoint / np.linalg.norm(midpoint)

                # Add to vertex list
                edge_midpoints[edge] = len(vertex_list)
                vertex_list.append(midpoint)

            return edge_midpoints[edge]

        # Subdivide each face into 4 triangles
        for face in faces:
            v0, v1, v2 = face

            # Get midpoints of edges
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)

            # Create 4 new triangular faces
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])

        faces = new_faces
        vertices = np.array(vertex_list)

    # Build graph from faces
    G = nx.Graph()

    # Add vertices as nodes
    for i, pos in enumerate(vertices):
        G.add_node(i, pos=tuple(pos))

    # Add edges from faces
    edges_set = set()
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges_set.add(edge)

    G.add_edges_from(edges_set)

    return G


def visualize(G, title="Polyhedral Sphere Wireframe"):
    """
    Visualize the polyhedral sphere wireframe in 3D.

    Args:
        G: NetworkX graph with 3D positions
        title: Plot title
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Extract coordinates
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]

    # Plot edges
    for edge in G.edges():
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.6, linewidth=1.0)

    # Plot nodes
    ax.scatter(xs, ys, zs, c='red', s=30, alpha=0.8, edgecolors='darkred', linewidth=0.5)

    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Make the plot look spherical
    max_range = 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.show()
