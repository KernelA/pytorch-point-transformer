from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

from .data_config import DataConfig


@dataclass
class TrainConfig:
    data: DataConfig = MISSING
    train_load_workers: int = MISSING
    test_load_workers: int = MISSING
    test_batch_size: int = MISSING
    train_batch_size: int = MISSING
    max_epochs: int = MISSING
    optimizer: Any = MISSING
    scheduler: Optional[Any] = None
    model: Any = MISSING
    log_dir: str = MISSING
    seed: int = MISSING
