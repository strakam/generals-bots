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

    def as_tensor(self, pad_to: int | None = None) -> np.ndarray:
        """
        Returns a 3D tensor of shape (15, rows, cols). Suitable for neural nets.
        """
        shape = self.armies.shape
        if pad_to is not None:
            shape = (pad_to, pad_to)
            assert pad_to >= max(self.armies.shape), "Can't pad to a smaller size than the original observation."
            # pad every channel with zeros, except for mountains, those are padded with ones
            h_pad = (0, pad_to - self.armies.shape[0])
            w_pad = (0, pad_to - self.armies.shape[1])
            self.armies = np.pad(self.armies, (h_pad, w_pad), "constant")
            self.generals = np.pad(self.generals, (h_pad, w_pad), "constant")
            self.cities = np.pad(self.cities, (h_pad, w_pad), "constant")
            self.mountains = np.pad(self.mountains, (h_pad, w_pad), "constant", constant_values=1)
            self.neutral_cells = np.pad(self.neutral_cells, (h_pad, w_pad), "constant")
            self.owned_cells = np.pad(self.owned_cells, (h_pad, w_pad), "constant")
            self.opponent_cells = np.pad(self.opponent_cells, (h_pad, w_pad), "constant")
            self.fog_cells = np.pad(self.fog_cells, (h_pad, w_pad), "constant")
            self.structures_in_fog = np.pad(self.structures_in_fog, (h_pad, w_pad), "constant")
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
