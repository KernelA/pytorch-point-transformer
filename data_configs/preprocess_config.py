from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class PreprocessConfig:
    num_points: int = MISSING
    include_normals: bool = MISSING
