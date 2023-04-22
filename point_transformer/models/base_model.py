from abc import abstractmethod
from typing import Union

import torch
from torch_geometric.data import Batch, Data

from ..types import PointSetBatchInfo


class BaseModel(torch.nn.Module):
    @torch.jit.unused
    def forward_data(self, data: Union[Data, Batch]):
        return self.forward((data.x, data.pos, data.batch))

    @torch.jit.unused
    def predict_class_data(self, data: Union[Data, Batch]) -> torch.Tensor:
        return self.predict_class(self.forward_data(data))

    @torch.jit.unused
    @abstractmethod
    def get_embedding(self, fpb_data: PointSetBatchInfo):
        pass

    @torch.jit.unused
    def get_embedding_data(self, data: Union[Data, Batch]) -> torch.Tensor:
        return self.get_embedding((data.x, data.pos, data.batch))

    def predict_class(self, predicted_logits: torch.Tensor) -> torch.Tensor:
        return predicted_logits.argmax(dim=-1)
