import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random


def generate(width=30, height=30, algorithm="dfs", seed=42):
    """
    Generate a maze graph using various maze generation algorithms.

    The maze is represented as a graph where nodes are cells and edges are
    passages between adjacent cells (no walls).

    Args:
        width: Width of the maze (number of cells)
        height: Height of the maze (number of cells)
        algorithm: Maze generation algorithm
                  - "dfs": Depth-first search (recursive backtracking)
                  - "prim": Randomized Prim's algorithm
                  - "kruskal": Randomized Kruskal's algorithm
                  - "wilson": Wilson's algorithm (uniform spanning tree)
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph representing the maze
    """
    np.random.seed(seed)
    random.seed(seed)

    G = nx.Graph()

    # Create all nodes (cells) with 2D positions
    node_id = 0
    coord_to_node = {}
    for y in range(height):
        for x in range(width):
            G.add_node(node_id, pos=(x, y), coords=(x, y))
            coord_to_node[(x, y)] = node_id
            node_id += 1

    def get_neighbors(x, y):
        """Get valid neighboring cells"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx_pos, ny_pos = x + dx, y + dy
            if 0 <= nx_pos < width and 0 <= ny_pos < height:
                neighbors.append((nx_pos, ny_pos))
        return neighbors

    if algorithm == "dfs":
        # Depth-first search (recursive backtracking)
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))

        while stack:
            x, y = stack[-1]

            # Get unvisited neighbors
            neighbors = get_neighbors(x, y)
            unvisited = [n for n in neighbors if n not in visited]

            if unvisited:
                # Choose random unvisited neighbor
                next_cell = random.choice(unvisited)
                # Add edge (remove wall)
                G.add_edge(coord_to_node[(x, y)], coord_to_node[next_cell])
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()

    elif algorithm == "prim":
        # Randomized Prim's algorithm
        visited = set()
        walls = []

        # Start from random cell
        start = (0, 0)
        visited.add(start)
        walls.extend([(start, n) for n in get_neighbors(*start)])

        while walls:
            # Pick random wall
            wall_idx = random.randint(0, len(walls) - 1)
            current, next_cell = walls.pop(wall_idx)

            if next_cell not in visited:
                # Add edge (remove wall)
                G.add_edge(coord_to_node[current], coord_to_node[next_cell])
                visited.add(next_cell)

                # Add neighboring walls
                for neighbor in get_neighbors(*next_cell):
                    if neighbor not in visited:
                        walls.append((next_cell, neighbor))

    elif algorithm == "kruskal":
        # Randomized Kruskal's algorithm (union-find)
        # Create all possible walls
        walls = []
        for y in range(height):
            for x in range(width):
                if x < width - 1:
                    walls.append(((x, y), (x + 1, y)))
                if y < height - 1:
                    walls.append(((x, y), (x, y + 1)))

        random.shuffle(walls)

        # Union-find data structure
        parent = {(x, y): (x, y) for y in range(height) for x in range(width)}

        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]

        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                parent[root2] = root1
                return True
            return False

        # Process walls
        for cell1, cell2 in walls:
            if union(cell1, cell2):
                G.add_edge(coord_to_node[cell1], coord_to_node[cell2])

    elif algorithm == "wilson":
        # Wilson's algorithm (loop-erased random walk)
        unvisited = {(x, y) for y in range(height) for x in range(width)}

        # Start with random cell in maze
        start = (0, 0)
        unvisited.remove(start)

        while unvisited:
            # Start random walk from random unvisited cell
            cell = random.choice(list(unvisited))
            path = [cell]

            # Random walk until we hit the maze
            while cell in unvisited:
                neighbors = get_neighbors(*cell)
                cell = random.choice(neighbors)

                # Loop erasure
                if cell in path:
                    # Erase loop
                    loop_start = path.index(cell)
                    path = path[:loop_start + 1]
                else:
                    path.append(cell)

            # Add path to maze
            for i in range(len(path) - 1):
                G.add_edge(coord_to_node[path[i]], coord_to_node[path[i + 1]])
                if path[i] in unvisited:
                    unvisited.remove(path[i])

    else:
        raise ValueError(f"Unknown maze algorithm: {algorithm}")

    return G


def visualize(G):
    """Visualize the maze graph showing walls and passages"""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Get node positions and determine maze dimensions
    pos = nx.get_node_attributes(G, 'pos')

    if not pos:
        print("No position data available for visualization")
        return

    # Determine maze dimensions
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width = max(xs) + 1
    height = max(ys) + 1

    # Draw the outer border
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', linewidth=3)

    # Draw walls (missing edges indicate walls)
    # For each cell, check if there's a wall on the right or bottom
    coords = nx.get_node_attributes(G, 'coords')

    for node, (x, y) in coords.items():
        # Check right wall
        if x < width - 1:
            right_neighbor = None
            for n in G.neighbors(node):
                if coords[n] == (x + 1, y):
                    right_neighbor = n
                    break
            if right_neighbor is None:
                # Draw vertical wall
                ax.plot([x + 1, x + 1], [y, y + 1], 'k-', linewidth=2)

        # Check bottom wall
        if y < height - 1:
            bottom_neighbor = None
            for n in G.neighbors(node):
                if coords[n] == (x, y + 1):
                    bottom_neighbor = n
                    break
            if bottom_neighbor is None:
                # Draw horizontal wall
                ax.plot([x, x + 1], [y + 1, y + 1], 'k-', linewidth=2)


    ax.set_xlim(-0.1, width + 0.1)
    ax.set_ylim(-0.1, height + 0.1)
    ax.set_aspect('equal')
    ax.set_title('Maze Graph')
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.show()
