from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    num_points: int = MISSING
    data_root: str = MISSING
    path_to_zip: str = MISSING
