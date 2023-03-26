
from torch import nn

from ..blocks.segmentation import (DownBlock, MLPBlock, PairedBlock,
                                   PairedBlockWithSkipConnection, UpBlock)
from ..layers.mlp import MLP
from ..types import PointSetBatchInfo
from .base_model import BaseModel


class PointSegmPointTransformer(BaseModel):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 num_neighs: int = 16,
                 num_paired_blocks_with_skip_connection: int = 1,
                 compress_dim_ratio_from_input: float = 1.0,
                 is_jit: bool = False):
        assert num_paired_blocks_with_skip_connection > 0

        super().__init__()

        out_features = 32

        self.point_segm_layer = nn.Linear(
            in_features=out_features,
            out_features=num_classes
        )

        in_features_paired = 2 * out_features

        self.paired_blocks = nn.ModuleList()

        self.paired_blocks.append(
            PairedBlockWithSkipConnection(
                MLPBlock(in_features=in_features,
                         out_features=out_features,
                         compress_dim=out_features // 2,
                         num_neighbors=num_neighs,
                         is_jit=is_jit),
                UpBlock(
                    in_features=in_features_paired,
                    in_features_original=out_features,
                    out_features=out_features,
                    compress_dim=out_features // 2,
                    num_neighbors=num_neighs,
                    is_jit=is_jit
                )
            )
        )

        num_paired_blocks_with_skip_connection -= 1

        intermediate_in_features = out_features

        for _ in range(num_paired_blocks_with_skip_connection):
            in_features_paired *= 2
            intermediate_in_features *= 2

            self.paired_blocks.append(
                PairedBlockWithSkipConnection(
                    DownBlock(
                        in_features=out_features,
                        out_features=intermediate_in_features,
                        compress_dim=intermediate_in_features // 2,
                        num_neighbors=num_neighs,
                        is_jit=is_jit),
                    UpBlock(
                        in_features=in_features_paired,
                        in_features_original=intermediate_in_features,
                        out_features=intermediate_in_features,
                        compress_dim=intermediate_in_features // 2,
                        num_neighbors=num_neighs,
                        is_jit=is_jit)
                )
            )

            out_features = intermediate_in_features

        intermediate_out_features = 2 * intermediate_in_features

        self.paired_blocks.append(
            PairedBlock(
                DownBlock(
                    in_features=intermediate_in_features,
                    out_features=intermediate_out_features,
                    compress_dim=intermediate_out_features // 2,
                    num_neighbors=num_neighs,
                    is_jit=is_jit
                ),
                MLPBlock(
                    in_features=intermediate_out_features,
                    out_features=intermediate_out_features,
                    compress_dim=intermediate_out_features // 2,
                    num_neighbors=num_neighs,
                    is_jit=is_jit
                )
            )
        )

    def get_embedding(self, fpb_data: PointSetBatchInfo):
        fpb_out = fpb_data

        # first pass half of the model
        for block in self.paired_blocks:
            fpb_out = block(fpb_out)

        # # seconds half of the model
        for block in reversed(self.paired_blocks):
            fpb_out = block(fpb_out)

        return fpb_out[0]

    def forward(self, fpb_data: PointSetBatchInfo):
        return self.point_segm_layer(self.get_embedding(fpb_data))
