import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate(major_radius=3, minor_radius=1, num_major=20, num_minor=12,
             separation_distance=None, array_size=20):
    """
    Create two torus wireframes that touch at exactly one vertex.

    The tori are positioned such that they share one vertex. The first torus is centered
    at the origin, and the second torus is positioned so that they touch.

    Each vertex stores a numpy array:
    - Torus 1: zeros everywhere except value 2 at the grid position
    - Torus 2: ones everywhere except value 2 at the grid position
    - Touching vertex: all values are 2

    Args:
        major_radius: Distance from center of torus to center of tube
        minor_radius: Radius of the tube
        num_major: Number of segments around the major circle
        num_minor: Number of segments around the minor circle
        separation_distance: Distance between centers. If None, calculated to make them touch.
        array_size: Size of the square numpy array to store at each vertex

    Returns:
        Combined NetworkX graph with both tori
    """
    # Generate first torus centered at origin
    G1, node_map1 = _generate_torus(major_radius, minor_radius, num_major, num_minor,
                                     center=(0, 0, 0))

    # Calculate separation distance if not provided
    if separation_distance is None:
        separation_distance = 2 * (major_radius + minor_radius)

    # Generate second torus, shifted along x-axis
    G2, node_map2 = _generate_torus(major_radius, minor_radius, num_major, num_minor,
                                     center=(separation_distance, 0, 0))

    # Find the touching vertices
    pos1 = nx.get_node_attributes(G1, 'pos')
    pos2 = nx.get_node_attributes(G2, 'pos')
    grid1 = nx.get_node_attributes(G1, 'grid_pos')
    grid2 = nx.get_node_attributes(G2, 'grid_pos')
    uv1 = nx.get_node_attributes(G1, 'uv')
    uv2 = nx.get_node_attributes(G2, 'uv')

    # Find the closest pair of vertices between the two tori
    min_dist = float('inf')
    touch_node1 = None
    touch_node2 = None

    for n1 in G1.nodes():
        for n2 in G2.nodes():
            p1 = np.array(pos1[n1])
            p2 = np.array(pos2[n2])
            dist = np.linalg.norm(p1 - p2)

            if dist < min_dist:
                min_dist = dist
                touch_node1 = n1
                touch_node2 = n2

    # Create combined graph
    G_combined = nx.Graph()

    # Add all nodes from G1 with their numpy arrays
    for node in G1.nodes():
        # Create array for torus 1: zeros except value 2 at grid position
        node_array = np.zeros((array_size, array_size), dtype=np.float32)
        i, j = grid1[node]
        node_array[i % array_size, j % array_size] = 2.0

        G_combined.add_node(f"T1_{node}",
                           pos=pos1[node],
                           torus=1,
                           grid_pos=grid1[node],
                           uv=uv1[node],
                           array=node_array)

    # Add all edges from G1
    for edge in G1.edges():
        G_combined.add_edge(f"T1_{edge[0]}", f"T1_{edge[1]}")

    # Add all nodes from G2, except the touching node
    for node in G2.nodes():
        if node == touch_node2:
            continue

        # Create array for torus 2: ones except value 2 at grid position
        node_array = np.ones((array_size, array_size), dtype=np.float32)
        i, j = grid2[node]
        node_array[i % array_size, j % array_size] = 2.0

        G_combined.add_node(f"T2_{node}",
                           pos=pos2[node],
                           torus=2,
                           grid_pos=grid2[node],
                           uv=uv2[node],
                           array=node_array)

    # Add all edges from G2, redirecting edges that involved the touching node
    for edge in G2.edges():
        n1, n2 = edge
        node1_name = f"T1_{touch_node1}" if n1 == touch_node2 else f"T2_{n1}"
        node2_name = f"T1_{touch_node1}" if n2 == touch_node2 else f"T2_{n2}"
        G_combined.add_edge(node1_name, node2_name)

    # Update the touching node
    if f"T1_{touch_node1}" in G_combined.nodes():
        G_combined.nodes[f"T1_{touch_node1}"]['torus'] = 'both'
        G_combined.nodes[f"T1_{touch_node1}"]['touching_point'] = True
        touching_array = np.full((array_size, array_size), 2.0, dtype=np.float32)
        G_combined.nodes[f"T1_{touch_node1}"]['array'] = touching_array

    # Store metadata
    G_combined.graph['touching_node'] = f"T1_{touch_node1}"
    G_combined.graph['touching_distance'] = min_dist

    return G_combined


def _generate_torus(major_radius=3, minor_radius=1, num_major=20, num_minor=12, center=(0, 0, 0)):
    """
    Generate a torus wireframe graph.

    Returns:
        NetworkX graph and node_map with grid indices
    """
    G = nx.Graph()
    cx, cy, cz = center

    node_map = {}
    node_id = 0

    for i in range(num_major):
        theta = 2 * np.pi * i / num_major

        for j in range(num_minor):
            phi = 2 * np.pi * j / num_minor

            # Parametric equations for torus
            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta) + cx
            y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta) + cy
            z = minor_radius * np.sin(phi) + cz

            # Calculate UV coordinates
            u = i / num_major
            v = j / num_minor

            G.add_node(node_id, pos=(x, y, z), grid_pos=(i, j), uv=(u, v))
            node_map[(i, j)] = node_id
            node_id += 1

    # Add edges
    for i in range(num_major):
        for j in range(num_minor):
            current = node_map[(i, j)]

            # Connect to next node in major circle
            next_major = node_map[((i + 1) % num_major, j)]
            G.add_edge(current, next_major)

            # Connect to next node in minor circle
            next_minor = node_map[(i, (j + 1) % num_minor)]
            G.add_edge(current, next_minor)

    return G, node_map


def visualize(G, title="Two Tori Touching at One Vertex"):
    """
    Visualize two touching tori wireframes in 3D.

    Args:
        G: NetworkX graph with 3D positions
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Get node positions and torus assignments
    pos = nx.get_node_attributes(G, 'pos')
    torus_attr = nx.get_node_attributes(G, 'torus')

    # Separate nodes by torus
    torus1_nodes = [n for n, t in torus_attr.items() if t == 1]
    torus2_nodes = [n for n, t in torus_attr.items() if t == 2]
    touching_nodes = [n for n, t in torus_attr.items() if t == 'both']

    # Plot edges
    for edge in G.edges():
        n1, n2 = edge
        t1 = torus_attr.get(n1, 1)
        t2 = torus_attr.get(n2, 1)

        if 'both' in [t1, t2]:
            color = 'purple'
            alpha = 0.8
            linewidth = 2.0
        elif t1 == 1 and t2 == 1:
            color = 'blue'
            alpha = 0.6
            linewidth = 1.0
        elif t1 == 2 and t2 == 2:
            color = 'green'
            alpha = 0.6
            linewidth = 1.0
        else:
            color = 'purple'
            alpha = 0.8
            linewidth = 2.0

        x_coords = [pos[n1][0], pos[n2][0]]
        y_coords = [pos[n1][1], pos[n2][1]]
        z_coords = [pos[n1][2], pos[n2][2]]
        ax.plot(x_coords, y_coords, z_coords, color=color, alpha=alpha, linewidth=linewidth)

    # Plot nodes for torus 1
    if torus1_nodes:
        xs = [pos[node][0] for node in torus1_nodes]
        ys = [pos[node][1] for node in torus1_nodes]
        zs = [pos[node][2] for node in torus1_nodes]
        ax.scatter(xs, ys, zs, c='blue', s=20, alpha=0.7, edgecolors='darkblue',
                   linewidth=0.5, label='Torus 1')

    # Plot nodes for torus 2
    if torus2_nodes:
        xs = [pos[node][0] for node in torus2_nodes]
        ys = [pos[node][1] for node in torus2_nodes]
        zs = [pos[node][2] for node in torus2_nodes]
        ax.scatter(xs, ys, zs, c='green', s=20, alpha=0.7, edgecolors='darkgreen',
                   linewidth=0.5, label='Torus 2')

    # Highlight the touching node
    if touching_nodes:
        xs = [pos[node][0] for node in touching_nodes]
        ys = [pos[node][1] for node in touching_nodes]
        zs = [pos[node][2] for node in touching_nodes]
        ax.scatter(xs, ys, zs, c='red', s=150, alpha=1.0, edgecolors='darkred',
                   linewidth=2.0, marker='*', label='Touching Point', zorder=100)

    # Calculate bounds for equal aspect ratio
    all_xs = [pos[node][0] for node in pos.keys()]
    all_ys = [pos[node][1] for node in pos.keys()]
    all_zs = [pos[node][2] for node in pos.keys()]

    x_range = max(all_xs) - min(all_xs)
    y_range = max(all_ys) - min(all_ys)
    z_range = max(all_zs) - min(all_zs)

    max_range = max(x_range, y_range, z_range)

    x_middle = (max(all_xs) + min(all_xs)) / 2
    y_middle = (max(all_ys) + min(all_ys)) / 2
    z_middle = (max(all_zs) + min(all_zs)) / 2

    # Set equal limits with some padding
    padding = max_range * 0.1
    ax.set_xlim(x_middle - max_range/2 - padding, x_middle + max_range/2 + padding)
    ax.set_ylim(y_middle - max_range/2 - padding, y_middle + max_range/2 + padding)
    ax.set_zlim(z_middle - max_range/2 - padding, z_middle + max_range/2 + padding)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set view angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()
