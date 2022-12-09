from typing import Optional

import torch
from torchmetrics import Metric


class AccMean(Metric):
    higher_is_better: Optional[bool] = False
    is_differentiable: Optional[bool] = False
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_loss", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("num_items", default=torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor, num_items: torch.LongTensor):
        self.sum_loss += loss
        self.num_items += num_items

    def compute(self):
        return self.sum_loss / self.num_items
