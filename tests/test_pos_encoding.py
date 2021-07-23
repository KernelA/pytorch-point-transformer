import pytest

import torch

from point_transformer import PositionalEncoder


def test_position_encoding():
    output_dim = 32
    layer = PositionalEncoder(output_dim=output_dim, hid_dim=16)
    point_positions = torch.ones((2, 5, 6))

    positional_encoding = layer(point_positions[..., :3], point_positions[..., 3:])

    assert positional_encoding.shape[-1] == output_dim
    assert positional_encoding.shape[1] == point_positions.shape[1]
