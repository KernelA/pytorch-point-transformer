from typing import Union

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from ..blocks import PointTransformerBlock, TransitionDown
from ..layers import MLP
from ..types import PointSetBatchInfo, TwoInputsType


class TupleInputSeq(nn.Sequential):
    """Only for TorchScript support
    """

    def forward(self, input: PointSetBatchInfo) -> PointSetBatchInfo:
        for module in self:
            input = module(input)
        return input


class ClsPointTransformer(nn.Module):
    def __init__(self, in_features: int,
                 num_classes: int,
                 num_neighs: int = 16,
                 num_transformer_blocks: int = 1,
                 compress_dim_ratio_from_input: float = 1.0,
                 is_jit: bool = False):
        super().__init__()
