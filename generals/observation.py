from dataclasses import dataclass
from numpy import ndarray
from typing import Dict

@dataclass
class Observation:
    army: ndarray
    general: ndarray
    city: ndarray
    owned_cells: ndarray
    opponent_cells: ndarray
    neutral_cells: ndarray
    visibile_cells: ndarray
    structure: ndarray
    action_mask: ndarray
    owned_land_count: int
    owned_army_count: int
    opponent_land_count: int
    opponent_army_count: int
    is_winner: bool
    timestep: int

    def as_dict(self) -> Dict[str, ndarray]:
        return self.__dict__
