from typing import Tuple

from torch import nn
import torch
from torch_geometric.nn import knn_interpolate

from ..types import PointSetBatchInfo

TwoInputsType = Tuple[torch.Tensor, torch.Tensor,
                      torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]


class TransitionUp(nn.Module):
    def __init__(self, *, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.ReLU()
        )
        self.linear_residual = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.ReLU()
        )

    def forward(self, input: TwoInputsType) -> PointSetBatchInfo:
        features_1, positions_1, batch_1, features_2, positions_2, batch_2 = input

        features_1 = self.linear(features_1)
        features_2 = self.linear_residual(features_2)

        interpolated_features = knn_interpolate(
            features_1, positions_1, positions_2, batch_x=batch_1, batch_y=batch_2)

        return interpolated_features + features_2, positions_2, batch_2
