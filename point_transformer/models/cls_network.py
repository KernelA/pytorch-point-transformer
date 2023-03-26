
from torch import nn
from torch_geometric.nn import global_mean_pool

from ..blocks import PointTransformerBlock, TransitionDown
from ..layers import MLP
from ..types import PointSetBatchInfo
from .base_model import BaseModel


class TupleInputSeq(nn.Sequential):
    """Only for TorchScript support
    """

    def forward(self, input: PointSetBatchInfo) -> PointSetBatchInfo:
        for module in self:
            input = module(input)
        return input


class ClsPointTransformer(BaseModel):
    def __init__(self, in_features: int,
                 num_classes: int,
                 num_neighs: int = 16,
                 num_transformer_blocks: int = 1,
                 compress_dim_ratio_from_input: float = 1.0,
                 is_jit: bool = False):
        super().__init__()
        out_features = 32

        transformer_blocks = [
            MLP(in_features, out_features),
            PointTransformerBlock(in_out_features=out_features,
                                  compress_dim=out_features // 2, num_neighbors=num_neighs, is_jit=is_jit)]

        classification_dim = 0

        for _ in range(1, num_transformer_blocks + 1):
            classification_dim = 2 * out_features
            transformer_blocks.extend(
                [TransitionDown(in_features=out_features,
                 out_features=classification_dim,
                                num_neighbors=num_neighs,
                                fps_sample_ratio=0.25),
                 PointTransformerBlock(in_out_features=classification_dim,
                                       compress_dim=round(
                                           compress_dim_ratio_from_input * out_features),
                                       num_neighbors=num_neighs,
                                       is_jit=is_jit)
                 ]
            )
            out_features = classification_dim

        self.feature_extractor = TupleInputSeq(*transformer_blocks)
        self.classification_head = nn.Linear(classification_dim, num_classes)

    def forward(self, fpb_data: PointSetBatchInfo):
        encoding = self.get_embedding(fpb_data)
        return self.classification_head(encoding)

    def get_embedding(self, fpb_data: PointSetBatchInfo):
        features, positions, batch = fpb_data
        assert positions.shape[1] == 3, "Expected 3D coordinates"
        new_features, _, new_batch = self.feature_extractor((features, positions, batch))
        return global_mean_pool(new_features, new_batch)
