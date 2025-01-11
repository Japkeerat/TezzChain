from typing import Optional
from dataclasses import dataclass


@dataclass
class BasicChunkConfig:
    max_characters: Optional[int] = None
    new_after_n_characters: Optional[int] = None
    overlap: Optional[int] = None
    overlap_all: Optional[bool] = None
