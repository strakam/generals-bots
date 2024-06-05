from typing import List, Tuple

from pydantic import BaseModel, validator, ValidationError

class GridConfig(BaseModel):
    grid_size: int = 32
    town_density: float = 0.1
    terrain_density: float = 0.1
    starting_positions: List[Tuple[int, int]] = [(0, 0), (10, 10)]
    map_name: str = None
