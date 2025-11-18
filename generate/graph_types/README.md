# Graph Types Module

This module contains modular graph generation and visualization functions. Each graph type is implemented in a separate file with two main functions: `generate()` and `visualize()`.

## Structure

```
graph_types/
├── __init__.py           # Module initialization
├── loader.py             # Dynamic loader for graph types
├── sphere.py             # Sphere wire mesh graph
├── grid.py               # 2D grid graph
├── torus.py              # N-dimensional torus graph
├── klein_bottle.py       # Klein bottle graph
├── maze.py               # Maze graphs (DFS, Prim, Kruskal, Wilson)
├── erdos_renyi.py        # Erdős-Rényi random graph
├── barabasi_albert.py    # Barabási-Albert scale-free graph
├── watts_strogatz.py     # Watts-Strogatz small-world graph
├── polyhedral_sphere.py  # Polyhedral sphere (icosphere, cube, octahedron)
└── touching_tori.py      # Two tori touching at one vertex
```

## Usage

### Using the Loader

The recommended way to use the graph types is through the loader module:

```python
from graph_types.loader import generate_graph, visualize_graph

# Generate a graph
graph = generate_graph('grid', width=20, height=20)

# Visualize it
visualize_graph('grid', graph)
```

### Direct Import

You can also import graph types directly:

```python
from graph_types import sphere

# Generate a sphere graph
graph = sphere.generate(num_horizontal=20, num_vertical=20)

# Visualize it
sphere.visualize(graph)
```

## Adding New Graph Types

To add a new graph type:

1. Create a new file in `graph_types/` (e.g., `my_graph.py`)
2. Implement two functions:
   - `generate(**kwargs)`: Returns a NetworkX graph
   - `visualize(G, **kwargs)`: Visualizes the graph using matplotlib
3. Add the module to `GRAPH_TYPE_MODULES` in `loader.py`:
   ```python
   GRAPH_TYPE_MODULES = {
       ...
       "my_graph": ".my_graph",
   }
   ```
4. Update the parameter extraction in `generate/generate_graph.py` if needed

## Graph Types Reference

### Sphere
**File:** `sphere.py`
**Parameters:** `num_horizontal`, `num_vertical`
**Description:** Wire mesh sphere with latitude and longitude circles

### Grid
**File:** `grid.py`
**Parameters:** `width`, `height`
**Description:** 2D rectangular grid graph

### Torus
**File:** `torus.py`
**Parameters:** `dimensions` (list of integers)
**Description:** N-dimensional torus with wraparound edges

### Klein Bottle
**File:** `klein_bottle.py`
**Parameters:** `u_points`, `v_points`
**Description:** Klein bottle mesh with non-orientable topology

### Maze
**File:** `maze.py`
**Parameters:** `width`, `height`, `algorithm`, `seed`
**Algorithms:** `dfs`, `prim`, `kruskal`, `wilson`
**Description:** Maze graphs generated using various algorithms

### Erdős-Rényi
**File:** `erdos_renyi.py`
**Parameters:** `n_nodes`, `edge_prob`, `seed`
**Description:** Random graph with independent edge probabilities

### Barabási-Albert
**File:** `barabasi_albert.py`
**Parameters:** `n_nodes`, `m_edges`, `seed`
**Description:** Scale-free network with preferential attachment

### Watts-Strogatz
**File:** `watts_strogatz.py`
**Parameters:** `n_nodes`, `k_neighbors`, `rewire_prob`, `seed`
**Description:** Small-world network with rewiring

### Polyhedral Sphere
**File:** `polyhedral_sphere.py`
**Parameters:** `sphere_type`, `subdivisions`, `num_latitude`, `num_longitude`
**Sphere Types:** `icosphere`, `uv_sphere`, `cube_sphere`, `octahedron`
**Description:** Polyhedral sphere meshes with various generation methods
- **icosphere**: Subdivided icosahedron (most uniform distribution)
- **uv_sphere**: UV sphere with latitude/longitude lines
- **cube_sphere**: Subdivided cube projected onto sphere
- **octahedron**: Subdivided octahedron

### Touching Tori
**File:** `touching_tori.py`
**Parameters:** `major_radius`, `minor_radius`, `num_major`, `num_minor`, `separation_distance`, `array_size`
**Description:** Two torus wireframes that touch at exactly one vertex
- Each vertex stores a numpy array with special values marking grid positions
- Torus 1: zeros everywhere except value 2 at grid position
- Torus 2: ones everywhere except value 2 at grid position
- Touching vertex: all values are 2
- Useful for testing graph algorithms on connected components
