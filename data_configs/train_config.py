from dataclasses import dataclass

from omegaconf import MISSING

from .data_config import DataConfig


@dataclass
class TrainConfig:
    data: DataConfig = MISSING
    train_batch_size: int = MISSING
    test_batch_size: int = MISSING
