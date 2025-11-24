import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import umap
import numpy as np


class EmbeddingArray(nn.Module):
    def __init__(self, array_size: int, latent_size: int):
        super().__init__()
        self.array_size = array_size
        self.latent_size = latent_size
        layers = [nn.Linear(array_size**2*3, latent_size),nn.ReLU()]
        for _ in range(3):
            layers.append(nn.Linear(latent_size, latent_size, bias=False))
            layers.append(nn.ReLU())
        self.embedding = nn.Sequential(*layers)
        #self.embedding = nn.Linear(array_size**2*3, latent_size, bias=False)


    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        return self.embedding(x)


class ContrastiveEmbeddingsModel(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        latent_size: int = 8,
        softplus_beta: float = 0.01,
        softplus_offset: float = 20.0,
        mult: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_size = vocab_size
        self.latent_size = latent_size
        self.softplus_beta = softplus_beta
        self.softplus_offset = softplus_offset
        self.mult = mult
        self.use_array = kwargs.get('use_array', False)
        if self.use_array:
            self.array_size = kwargs.get('array_size', None)
            self.node_embedding = EmbeddingArray(self.array_size, self.latent_size)
        else:
            self.node_embedding = nn.Embedding(self.input_size, self.latent_size)
        # Replace simple embedding with convolutional embedding

    def forward(self, batch):
        # x shape: (batch_size, seq_len, height, width) for array-based inputs
        # We need to process each position in the sequence separately
        x = batch['input_ids']
        batch_size, seq_len = x.shape[:2]

        # Reshape to process all positions at once
        # x: (batch_size, seq_len, height, width) -> (batch_size * seq_len, height, width)
        #x_reshaped = x.view(batch_size * seq_len, *x.shape[2:])

        # Get embeddings for all positions
        embs = self.node_embedding(x)  # (batch_size * seq_len, latent_size)

        # Reshape back to (batch_size, seq_len, latent_size)
        #mbs = embs_flat.view(batch_size, seq_len, self.latent_size)

        emb1 = embs[:, 0]
        emb2 = embs[:, 1]
        # lc_loss, lc_distance, lc_sq_dev, lc_violation, lc_lagrange_mult = self.local_constraint_loss(emb1, emb2)
        emb3 = torch.roll(emb2, 1, dims=0)  # create random pairs of zx and zy
        result = {"emb1": emb1, "emb2": emb2, "emb3": emb3, "logits": None}
        return result

    def get_loss(self, result, batch):
        emb1 = result['emb1']
        emb2 = result['emb2']
        emb3 = result['emb3']

        # Compute contrastive loss: embeddings of adjacent nodes should be close
        distances = torch.norm(emb1 - emb2, p=1.8, dim=-1)
        sq_deviation = (distances - 1).square().mean()
        loss = sq_deviation

        # Contrastive part: embeddings of non-adjacent nodes should be far
        def smooth_id_log(x):
            threshold = 5
            # For x <= 10: output x
            # For x > 10: output 10 + log(1 + (x-10)), smoothly transitions at 10
            return torch.where(
                x <= threshold,
                x,
                threshold + torch.log1p(x - threshold)
            )
        raw_distances = torch.norm(emb1 - emb3, p=2, dim=-1)
        distances = smooth_id_log(raw_distances).mean()
        #distances = F.softplus(self.softplus_offset - distances, beta=self.softplus_beta)
        loss = loss - distances * self.mult

        result["loss"] = loss
        return result

    def get_param_groups(self):
        """Return a single parameter group with all model parameters."""
        param_groups = [{
            'params': list(self.parameters()),
        }]
        return param_groups

    def get_embeddings(self, graph, num_vertices):
        """
        Obtain PCA-projected node embeddings for all vertices in the graph from their array representations.

        Args:
            model: The model with ConvNodeEmbedding
            graph: NetworkX graph with 'array' attribute on each node
            device: Device to run computation on ('cpu' or 'cuda')

        Returns:
            embeddings_3d: numpy array of shape (num_nodes, 3)
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():

            if self.use_array:
                node_arrays = [graph.nodes[node]['array'] for node in graph.nodes()]
                node_arrays = [np.eye(3, dtype=np.float32)[np.array(node_array, dtype=np.int64)] for node_array in node_arrays]
                node_arrays_tensor = torch.tensor(node_arrays, dtype=torch.float).to(device).unsqueeze(1)
                embeddings = self.node_embedding(node_arrays_tensor).squeeze(1)
            else:       
                # Extract all node indices from the graph
                node_ids = sorted(graph.nodes())  # Sort to maintain consistent ordering

                # For this architecture, the node index itself is sufficient for node_embedding()
                node_arrays = [i for i, node_id in enumerate(node_ids)]

                # Stack into a batch tensor: (num_nodes)
                node_arrays_tensor = torch.tensor(node_arrays, dtype=torch.long).to(device)

                # Get embeddings through the convolutional network
                embeddings = self.node_embedding(node_arrays_tensor)  # (num_nodes, latent_size)

            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()

        # Project embeddings to 3D using UMAP

        return embeddings_np
