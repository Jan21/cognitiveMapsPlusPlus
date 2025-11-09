# CognitiveMaps++
This repository contains experiments testing different architectures for prediction of shortest paths on a graph given a starting and a goal state. The goal is to provide insights into how different architectures embed states in complex state-spaces into a high-dimensional space in order to enable effective navigation. Such embeddings are idealistic models of Cognitive Maps studied in neuroscience.


The following visualization shows an evolution of PCA projected node embeddings trained on random trajectories on a 2D mesh of a two connected 2D tori that share one point. The embeddings are optimized using a simple loss that penalizes distances of neighbor nodes to be different that 1 and also tries to pull the embeddings of random nodes as far as possible from each other. As can be seen, this simple optimization objective leads to a geometry which reflects underlying graph. Each torus is colored by a different color and the shades correspond to the coordinates in the given torus. At the end, the animation also shows what happens if we turn off the repulsive term of the loss function.

<p align="center">
  <img src="assets/embedding_evolution.gif" alt="Embedding evolution" width="800"/>
</p>

The visualization above demonstrates two interesting facts. First, the embeddedings form a stratiefied space and not a smooth manifold as frequently discussed in Machine Learning literature (the touching point between the two tori has a higher dimension than the rest of the points). Second, the distances between points are not well represented by a line segments which is the common assumption in goal-conditioned RL which uses metric or quasimetric embeddings to estimate a value function.


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
There are multiple graphs with configurable parameters. The graph type can be set in `config/config.yaml` in the variable `data`. The parameters of the corresponding file can be set in `config/data/$GRAPH_TYPE$.yaml`.

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
From the graph generated in step 1, we can generate training and testing data. The `config.yaml` file contains parameters to determine minimum and maximum path length and also its suboptimality. The suboptimal paths are generated from the optimal ones by subdividing them into multiple segments and then perturbing each segment. The resulting paths will be saved in JSON format in the `temp` directory.

## 3. Model Training
The model to train can be set in the `config/config.yaml` file. Currently, there are two choices: `transformer` (a simple autoregressive transformer) and `diffusion_upsample` (a custom diffusion model which upscales the output during generation).


## 4. Visualizing the Trained Node Embeddings
To visualize the embeddings of a given model, the checkpoint needs to be specified and the proper architecture needs to be set in `config/config.yaml`. Other arguments enable setting the projection method (UMAP vs PCA) and dimension (2D vs 3D).
