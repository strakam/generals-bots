import dataclasses
from typing import Any, TypeAlias

import numpy as np

# Type aliases
Info: TypeAlias = dict[str, Any]


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

    def to_tensor(self, pad_to: int = None) -> np.ndarray:
        """
        Returns a uniformly sized tensor with shape: (15, height, width).
        Scalar fields such as timestep or opponent_land_count are
        upsampled to the size of the grid.

        Optionally increases the size of each channel to pad_to by effectively
        filling the lower-right corner of the game-board with mountains.
        The returned shape becomes: (15, pad_to, pad_to).
        """

        grid_dims = self.armies.shape

        if pad_to is not None:
            assert pad_to >= max(grid_dims), "Can't pad to a size smaller than the original observation."
            # Numpy expects a tuple representing the amount to pad each dimension before the
            # data already in the array & after that data.
            height_padding = (0, pad_to - grid_dims[0])
            width_padding = (0, pad_to - grid_dims[1])
            grid_dims = (pad_to, pad_to)

        else:
            height_padding = (0, 0)
            width_padding = (0, 0)

        channels = []
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
                channel = channel * np.ones(shape=grid_dims)
            else:
                raise Exception(f"Unable to appropriately process channel: {channel}.")

            channels.append(channel)

        return np.stack(channels, axis=0)
