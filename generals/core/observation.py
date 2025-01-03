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

    def values(self):
        return dataclasses.asdict(self).values()

    def items(self):
        return dataclasses.asdict(self).items()

    def as_tensor(self):
        """
        Returns a 3D tensor of shape (15, rows, cols). Suitable for neural nets.
        """
        shape = self.armies.shape
        return np.stack(
            [
                self.armies,
                self.generals,
                self.cities,
                self.mountains,
                self.neutral_cells,
                self.owned_cells,
                self.opponent_cells,
                self.fog_cells,
                self.structures_in_fog,
                np.ones(shape) * self.owned_land_count,
                np.ones(shape) * self.owned_army_count,
                np.ones(shape) * self.opponent_land_count,
                np.ones(shape) * self.opponent_army_count,
                np.ones(shape) * self.timestep,
                np.ones(shape) * self.priority,
            ],
            axis=0,
        )
