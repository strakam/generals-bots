import dataclasses

import numpy as np


@dataclasses.dataclass
class Observation(dict):
    """
    We override some dictionary methods and subclass dict to allow the
    Observation object to be accessible in dictionary-style format,
    e.g. observation["armies"]. And to allow for providing a
    listing of the keys/attributes.

    These steps are necessary because PettingZoo & Gymnasium expect
    dictionary-like Observation objects, but we want the benefits of
    knowing the dictionaries' members which a dataclass/class provides.
    """

    armies: np.ndarray
    generals: np.ndarray
    cities: np.ndarray
    mountains: np.ndarray
    neutral_cells: np.ndarray
    owned_cells: np.ndarray
    opponent_cells: np.ndarray
    fog_cells: np.ndarray
    structures_in_fog: np.ndarray
    owned_land_count: int
    owned_army_count: int
    opponent_land_count: int
    opponent_army_count: int
    timestep: int
    priority: int = 0

    def __getitem__(self, attribute_name: str):
        return getattr(self, attribute_name)

    def keys(self):
        return dataclasses.asdict(self).keys()
