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

    def pad_observation(self, pad_to: int) -> None:
        """
        Pads all the observation arrays to the specified size.

        Args:
            pad_to (int): The target size to pad to. Must be >= the current observation size.
        """
        assert pad_to >= max(self.armies.shape), "Can't pad to a smaller size than the original observation."

        h_pad = (0, pad_to - self.armies.shape[0])
        w_pad = (0, pad_to - self.armies.shape[1])

        # Regular zero padding for most arrays
        zero_pad_arrays = [
            "armies",
            "generals",
            "cities",
            "neutral_cells",
            "owned_cells",
            "opponent_cells",
            "fog_cells",
            "structures_in_fog",
        ]

        for array_name in zero_pad_arrays:
            setattr(self, array_name, np.pad(getattr(self, array_name), (h_pad, w_pad), "constant"))

        # Special case for mountains which are padded with ones
        self.mountains = np.pad(self.mountains, (h_pad, w_pad), "constant", constant_values=1)

    def as_tensor(self, pad_to: int | None = None) -> np.ndarray:
        """
        Returns a 3D tensor of shape (15, rows, cols). Suitable for neural nets.
        """
        if pad_to is not None:
            self.pad_observation(pad_to)
            shape = (pad_to, pad_to)
        else:
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
