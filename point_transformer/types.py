from typing import Tuple

import torch

PointSetBatchInfo = Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]

TwoInputsType = Tuple[torch.Tensor,
                      torch.Tensor,
                      torch.LongTensor,
                      torch.Tensor,
                      torch.Tensor,
                      torch.LongTensor]
