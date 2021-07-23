from torch import nn
from torch.nn import functional as F


class LinearBN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_mapping = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, input):
        """input [B x N x in_features]
        """
        linear_mapping = self.linear_mapping(input)
        normalized = self.bn(linear_mapping.permute(0, 2, 1)).permute(0, 2, 1)
        return F.relu(normalized)
