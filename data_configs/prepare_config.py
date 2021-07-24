from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class PrepareConfig:
    path_to_archive: str = MISSING
    path_to_out_dir: str = MISSING
