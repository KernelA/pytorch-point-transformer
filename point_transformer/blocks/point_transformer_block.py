import torch
from torch import nn

from ..layers import PointTransformerLayer
from ..types import PointSetBatchInfo


class PointTransformerBlock(nn.Module):
    def __init__(self, *, in_out_features: int, compress_dim: int, num_neighbors: int, is_jit: bool = False):
        super().__init__()
        self.linear_encoder = nn.Linear(in_out_features, compress_dim)
        self.transformer = PointTransformerLayer(
            in_features=compress_dim,
            out_features=compress_dim,
            num_neighbors=num_neighbors)

        if is_jit:
            self.transformer = self.transformer.jittable()

        self.linear_decoder = nn.Linear(compress_dim, in_out_features)

    def forward(self, fpb_data: PointSetBatchInfo) -> PointSetBatchInfo:
        """input contains:
            features [B x N x in_features] - node features
            positions [B x N x num_coords] - position of points. By default num_coords is equal to 3.
            batch - batch indices
        """
        features, positions, batch = fpb_data

        encoded_features = self.linear_encoder(features)
        compressed_features, positions, batch = self.transformer(
            (encoded_features, positions, batch))
        output_features = self.linear_decoder(compressed_features)
        return features + output_features, positions, batch
