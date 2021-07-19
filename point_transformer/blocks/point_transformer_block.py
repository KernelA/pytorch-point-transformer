from torch import nn

from ..layers import PointTransformerLayer


class PointTransformerBlock(nn.Module):
    def __init__(self, *, in_features: int, compress_dim: int, num_neighbors: int):
        super().__init__()
        self.linear_encoder = nn.Linear(in_features, compress_dim)
        self.transformer = PointTransformerLayer(
            in_features=compress_dim, out_features=compress_dim, num_neighbors=num_neighbors)
        self.linear_decoder = nn.Linear(compress_dim, in_features)

    def forward(self, features, positions, batch):
        encoded_features = self.linear_encoder(features)
        output_features = self.linear_decoder(self.transformer(encoded_features, positions, batch))
        return features + output_features
