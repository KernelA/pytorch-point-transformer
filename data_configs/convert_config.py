from dataclasses import dataclass
from typing import List

from omegaconf import MISSING


@dataclass
class ConvertConfig:
    path_to_archive: str = MISSING
    path_to_hdf5: str = MISSING
    extensions: List[str] = MISSING
    mesh_format: str = MISSING
