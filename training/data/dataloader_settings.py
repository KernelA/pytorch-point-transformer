from dataclasses import dataclass
from typing import Any


@dataclass
class LoadSettings:
    num_workers: int
    batch_size: int
    pre_transform: Any = None
    pre_filter: Any = None
    transform: Any = None
