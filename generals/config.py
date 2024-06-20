from typing import List, Tuple, Literal, Dict

from pydantic import BaseModel

class Config(BaseModel):

    # Game settings
    n_players: int = 2
    grid_size: int = 16
    town_density: float = 0.1
    mountain_density: float = 0.1
    map_name: str = None
    increment_rate: int = 50
