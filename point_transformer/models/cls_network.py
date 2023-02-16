from typing import Union

from torch import nn
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch, Data

from ..blocks import PointTransformerBlock, TransitionDown
from ..types import PointSetBatchInfo


class TupleInputSeq(nn.Sequential):
    """Only for TorchScript support
    """

    def forward(self, input: PointSetBatchInfo) -> PointSetBatchInfo:
        for module in self:
            input = module(input)
        return input


class InitMapping(nn.Module):
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


class ClsPointTransformer(nn.Module):
    def __init__(self, in_features: int,
                 num_classes: int,
                 num_neighs: int = 16,
                 num_transformer_blocks: int = 1,
                 compress_dim_ratio_from_input: float = 1.0,
                 is_jit: bool = False):
        super().__init__()
        out_features = 32

        transformer_blocks = [
            InitMapping(in_features, out_features),
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

    @torch.jit.unused
    def forward_data(self, data: Union[Data, Batch]):
        return self.forward((data.x, data.pos, data.batch))

    @torch.jit.unused
    def predict_class_data(self, data: Union[Data, Batch]) -> torch.Tensor:
        return self.predict_class(self.forward_data(data))

    @torch.jit.unused
    def get_embedding_data(self, data: Union[Data, Batch]) -> torch.Tensor:
        return self.get_embedding((data.x, data.pos, data.batch))

    def predict_class(self, predicted_logits: torch.Tensor) -> torch.Tensor:
        return predicted_logits.argmax(dim=-1)
