#!/usr/bin/env python
"""Test script to generate and visualize all graph types sequentially."""

import sys
sys.path.insert(0, '..')

from graph_types.loader import generate_graph, visualize_graph, GRAPH_TYPE_MODULES


def test_all_graph_types():
    """Generate and visualize each graph type. Close the window to see the next one."""

    # Define test parameters for each graph type
    test_configs = {
        "sphere": {
            "num_horizontal": 10,
            "num_vertical": 12
        },
        "grid": {
            "width": 15,
            "height": 15
        },
        "torus": {
            "dimensions": [8, 8]
        },
        "klein_bottle": {
            "u_points": 20,
            "v_points": 20
        },
        "maze": {
            "width": 20,
            "height": 20,
            "algorithm": "dfs",
            "seed": 42
        },
        "erdos_renyi": {
            "n_nodes": 50,
            "edge_prob": 0.1,
            "seed": 42
        },
        "barabasi_albert": {
            "n_nodes": 50,
            "m_edges": 3,
            "seed": 42
        },
        "watts_strogatz": {
            "n_nodes": 50,
            "k_neighbors": 6,
            "rewire_prob": 0.3,
            "seed": 42
        },
        "polyhedral_sphere": {
            "sphere_type": "icosphere",
            "subdivisions": 2,
            "num_latitude": 16,
            "num_longitude": 20
        },
        "touching_tori": {
            "major_radius": 3,
            "minor_radius": 1,
            "num_major": 15,
            "num_minor": 12,
            "array_size": 20
        }
    }

    print("=" * 70)
    print("Testing All Graph Types")
    print("=" * 70)
    print(f"Total graph types to test: {len(test_configs)}")
    print("Close each visualization window to proceed to the next graph type.\n")

    for idx, (graph_type, params) in enumerate(test_configs.items(), 1):
        print(f"\n[{idx}/{len(test_configs)}] Testing: {graph_type.upper()}")
        print("-" * 70)
        print(f"Parameters: {params}")

        try:
            # Generate the graph
            print(f"Generating {graph_type} graph...")
            graph = generate_graph(graph_type, **params)

            # Print basic stats
            print(f"✓ Graph generated successfully!")
            print(f"  - Nodes: {graph.number_of_nodes()}")
            print(f"  - Edges: {graph.number_of_edges()}")
            print(f"  - Average degree: {2 * graph.number_of_edges() / graph.number_of_nodes():.2f}")

            # Visualize
            print(f"Visualizing {graph_type} graph...")
            print(">>> Close the window to continue to the next graph type <<<")
            visualize_graph(graph_type, graph)

            print(f"✓ {graph_type} visualization closed.")

        except Exception as e:
            print(f"✗ Error with {graph_type}: {e}")
            import traceback
            traceback.print_exc()
            response = input("Continue to next graph type? (y/n): ")
            if response.lower() != 'y':
                break

    print("\n" + "=" * 70)
    print("All graph types tested!")
    print("=" * 70)


if __name__ == "__main__":
    test_all_graph_types()
