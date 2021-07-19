from typing import Tuple

from torch import nn
import torch

from ..layers import PointTransformerLayer
from ..types import PointSetBatchInfo


class PointTransformerBlock(nn.Module):
    def __init__(self, *, in_out_features: int, compress_dim: int, num_neighbors: int):
        super().__init__()
        self.linear_encoder = nn.Linear(in_out_features, compress_dim)
        self.transformer = PointTransformerLayer(
            in_features=compress_dim, out_features=compress_dim, num_neighbors=num_neighbors)
        self.linear_decoder = nn.Linear(compress_dim, in_out_features)

    def forward(self, input: PointSetBatchInfo) -> PointSetBatchInfo:
        """input contains:
            features [N x in_features] - node features
            positions [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch - batch indices
        """
        features, positions, batch = input
        encoded_features = self.linear_encoder(features)
        compressed_features, positions, batch = self.transformer(
            (encoded_features, positions, batch))
        output_features = self.linear_decoder(compressed_features)
        return features + output_features, positions, batch
