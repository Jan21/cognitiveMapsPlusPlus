# CognitiveMaps++
This repository contains experiments testing different architectures for prediction of shortest paths on a graph given starting and a goal state. The goal is to provide insights of how different architectures embed state in complex state-spaces into a high-dimensional space in order to enable effective navigation. 

To reproduce the results in this repository, set the parameters in `config/config.yaml` and follow these steps:


1. **Generate the graph**:
   ```
   python generate/generate_graph.py
   ```

2. **Sample training and validation data**:    
   ```
   python generate/generate_graph.py data=maze
   ```

3. **Train the model**: 
   ```
   python train.py
   ```

4. **Visualize the embeddings**: 
   ```
   python visualize/visualize_embeddings.py --checkpoint="PATH_TO_A_CHECKPOINT"
   ```


## 1. Graph Generation
There are multiple graph with configurable parameters. The graph type can be set in `config/config.yaml` in the variable `data`. The parametrs of the corresponding file can be set in `config/data/$GRAPH_TYPE$.yaml`.

### Available Graph Types

| Graph Type | Config string |
|------------|---------|
| 2D Grid | `grid` |
| Sphere Wire Mesh | `sphere` |
| n-Dimensional Torus | `nd_torus` |
| Klein Bottle | `klein_bottle` |
| Maze | `maze` |
| Erdős-Rényi Random | `erdos_renyi` |
| Barabási-Albert Scale-Free | `barabasi_albert` |
| Watts-Strogatz Small-World | `watts_strogatz` |


## 2. Data Generation
From the graph generated in step 1, we can generate training and testing data. The `config.yaml` file contains parameters to determine minimum and maximum path lenght and also its suboptimality. The suboptimal paths are generated from the optimal ones by subdividing them to multiple segments and then perturbing each segment. The resulting paths will be saves in json format in the `temp` directory.

## 3. Model training
The model to train can be set in the `config/congig.yaml` file. Currently, there are two choices: `transformer` (a simple autoregressive transformer) and `diffusion_upsample` (a custom diffusion model which is upscaling the output during generation).


## 4. Visualizing the trained node embeddings
To visualize the the embeddings of a given model, the checkpoint needs to be specified and the proper architecture needs to be set in `config/congig.yaml`. Other arguments enable to set the projection method (UMAP vs PCA) and dimension (2D vs 3D).
