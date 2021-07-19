from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, knn_graph

from .positional_encoding import PositionalEncoder


class PointTransformerLayer(MessagePassing):
    def __init__(self, *,
                 in_features: int,
                 out_features: int,
                 num_neighbors: int,
                 gamma_mlp_hidden_dim: int = 32,
                 pos_encoder_hidden_dim: int = 32,
                 num_coords: int = 3):
        super().__init__(aggr="add")
        self.positional_encoder = PositionalEncoder(
            num_coords=num_coords, hid_dim=pos_encoder_hidden_dim, output_dim=out_features)
        self.phi = nn.Linear(in_features, out_features)
        self.psi = nn.Linear(in_features, out_features)
        self.alpha = nn.Linear(in_features, out_features)
        self.gamma = nn.Sequential(nn.Linear(out_features, gamma_mlp_hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(gamma_mlp_hidden_dim, out_features)
                                   )
        self.knn_num_neighs = num_neighbors

    def forward(self, features, positions, batch):
        """features [N x in_features] - node features
           positions [N x num_coords] - position of points. By default num_coords is equal to 3.
           batch - batch indices
        """
        edge_indices = knn_graph(features, k=self.knn_num_neighs, batch=batch)
        return self.propagate(edge_indices, x=features, pos=positions)

    def message(self, x_i, x_j, pos_i, pos_j):
        pos_encoding = self.positional_encoder(pos_i, pos_j)
        return F.softmax(self.gamma(self.phi(x_i) - self.psi(x_j) + pos_encoding),
                         dim=-1) * (self.alpha(x_j) + pos_encoding)
