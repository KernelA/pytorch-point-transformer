from typing import Tuple

from torch import nn
import torch
from torch_geometric.nn import knn_interpolate

from ..types import PointSetBatchInfo
from ..layers import LinearBN

TwoInputsType = Tuple[torch.Tensor,
                      torch.Tensor,
                      torch.LongTensor,
                      torch.Tensor,
                      torch.Tensor,
                      torch.LongTensor]


class TransitionUp(nn.Module):
    def __init__(self, *, in_features: int, out_features: int):
        super().__init__()
        self.linear = LinearBN(in_features, out_features)
        self.linear_residual = LinearBN(in_features, out_features)

    def forward(self, input: TwoInputsType) -> PointSetBatchInfo:
        """input contains:
            features_1 [B x N x in_features] - node features
            positions_1 [B x N x num_coords] - position of points. By default num_coords is equal to 3.
            batch_1 - batch indices

            features_2 [B x N x in_features] - node features
            positions_2 [B x N x num_coords] - position of points. By default num_coords is equal to 3.
            batch_2 - batch indices
        """
        features_1, positions_1, batch_1, features_2, positions_2, batch_2 = input

        features_1 = self.linear(features_1)
        features_2 = self.linear_residual(features_2)

        batch_size = features_1.shape[0]
        num_features = features_1.shape[-1]
        num_coords = positions_1.shape[-1]

        flatten_features_1 = features_1.reshape(-1, num_features)
        flatten_positions_1 = positions_1.view(-1, num_coords)
        flatten_positions_2 = positions_2.view(-1, num_coords)

        interpolated_features = knn_interpolate(
            flatten_features_1, flatten_positions_1, flatten_positions_2, batch_x=batch_1, batch_y=batch_2)

        interpolated_features = interpolated_features.view(batch_size, -1, num_features)

        return interpolated_features + features_2, positions_2, batch_2
