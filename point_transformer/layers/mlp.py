from torch import nn

from ..types import PointSetBatchInfo


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._mapping = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Linear(out_features, out_features, bias=False),
        )

    def forward(self, input_data: PointSetBatchInfo) -> PointSetBatchInfo:
        features, positions, batch = input_data
        return self._mapping(features), positions, batch
