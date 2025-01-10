from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass
class Observation(np.ndarray, Mapping):
    """
    Observation dataclass is by default a 15-channel 3D grid.
    This is due to its convenience when using ML methods (e.g. CNNs) and vectorized environments.
    However, you can also access respective channels as attributes, by their name, e.g.:

    obs = Observation(...)
    print(obs.armies)

    channel_names = obs.keys()
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

    def __new__(cls, *args, **kwargs):
        ones = np.ones(kwargs["armies"].shape, dtype=int)

        stacked = np.stack(
            [
                kwargs["armies"],
                kwargs["generals"],
                kwargs["cities"],
                kwargs["mountains"],
                kwargs["neutral_cells"],
                kwargs["owned_cells"],
                kwargs["opponent_cells"],
                kwargs["fog_cells"],
                kwargs["structures_in_fog"],
                kwargs["owned_land_count"] * ones,
                kwargs["owned_army_count"] * ones,
                kwargs["opponent_land_count"] * ones,
                kwargs["opponent_army_count"] * ones,
                kwargs["timestep"] * ones,
                kwargs["priority"] * ones,
            ]
        )
        obj = super().__new__(cls, shape=stacked.shape, dtype=stacked.dtype, buffer=stacked)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dataclass_fields__.keys())

    def __len__(self):
        return len(self.__dataclass_fields__)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return list(self.__dataclass_fields__.keys())

    # should return padded tensor
    def pad_observation(self, pad_to: int) -> "Observation":
        """
        Pads the observation to a square grid of size pad_to.
        """
        assert pad_to >= max(self.armies.shape), "Can't pad to a size smaller than the original observation."

        height_padding = (0, pad_to - self.armies.shape[0])
        width_padding = (0, pad_to - self.armies.shape[1])

        for attribute_name in self.keys():
            channel = self[attribute_name]
            is_arr = isinstance(channel, np.ndarray)
            is_mountain_channel = attribute_name == "mountains"
            is_scalar = isinstance(channel, int | np.integer)

            if is_arr and not is_mountain_channel:
                channel = np.pad(channel, (height_padding, width_padding), mode="constant", constant_values=0)
            elif is_arr and is_mountain_channel:
                channel = np.pad(channel, (height_padding, width_padding), mode="constant", constant_values=1)
            elif is_scalar:
                channel = channel * np.ones(shape=(pad_to, pad_to))
            else:
                raise Exception(f"Unable to appropriately process channel: {channel}.")

            setattr(self, attribute_name, channel)
        return self
