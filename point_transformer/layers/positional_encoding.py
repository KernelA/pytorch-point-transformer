from torch import nn
import torch


class PositionalEncoder(nn.Module):
    def __init__(self, *, output_dim: int, hid_dim: int, num_coords: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_coords, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, output_dim)
        )

    def forward(self, pos_i: torch.Tensor, pos_j: torch.Tensor) -> torch.Tensor:
        assert pos_i.shape[-1] == pos_j.shape[-1]
        """Positional encoding

        pos_i: [B x N x num_coords]
        pos_j: [B x N x num_coords]
        """
        return self.mlp(pos_i - pos_j)
