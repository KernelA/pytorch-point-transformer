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
        batch_size = input.shape[0]
        num_samples = input.shape[1]
        linear_mapping = self.linear_mapping(input)
        normalized = self.bn(linear_mapping.view(-1, self.bn.num_features)
                             ).view(batch_size, num_samples, self.bn.num_features)
        return F.relu(normalized)
