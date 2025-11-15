"""Graph type loader module to dynamically import graph generation and visualization functions."""

import importlib


# Mapping of graph types to their module names
GRAPH_TYPE_MODULES = {
    "sphere": ".sphere",
    "grid": ".grid",
    "torus": ".torus",
    "klein_bottle": ".klein_bottle",
    "maze": ".maze",
    "erdos_renyi": ".erdos_renyi",
    "barabasi_albert": ".barabasi_albert",
    "watts_strogatz": ".watts_strogatz",
}


def get_graph_module(graph_type):
    """
    Load the graph module for the specified graph type.

    Args:
        graph_type: String identifier for the graph type

    Returns:
        Module containing generate() and visualize() functions

    Raises:
        ValueError: If graph type is not supported
    """
    if graph_type not in GRAPH_TYPE_MODULES:
        supported = ", ".join(f"'{k}'" for k in GRAPH_TYPE_MODULES.keys())
        raise ValueError(
            f"Unknown graph type: '{graph_type}'. "
            f"Supported types: {supported}"
        )

    module_name = GRAPH_TYPE_MODULES[graph_type]
    return importlib.import_module(module_name, package=__package__)


def generate_graph(graph_type, **kwargs):
    """
    Generate a graph of the specified type.

    Args:
        graph_type: String identifier for the graph type
        **kwargs: Parameters to pass to the generation function

    Returns:
        NetworkX graph
    """
    module = get_graph_module(graph_type)
    return module.generate(**kwargs)


def visualize_graph(graph_type, graph, **kwargs):
    """
    Visualize a graph of the specified type.

    Args:
        graph_type: String identifier for the graph type
        graph: NetworkX graph to visualize
        **kwargs: Additional parameters to pass to the visualization function
    """
    module = get_graph_module(graph_type)
    return module.visualize(graph, **kwargs)
