import numpy as np
from scipy.ndimage import maximum_filter  # type: ignore

from .config import MOUNTAIN, PASSABLE

valid_generals = ["A", "B"]  # Generals are represented by A and B


class Channels:
    """
    armies - army size in each cell
    generals - general mask (1 if general is in cell, 0 otherwise)
    mountains - mountain mask (1 if cell is mountain, 0 otherwise)
    cities - city mask (1 if cell is city, 0 otherwise)
    passable - passable mask (1 if cell is passable, 0 otherwise)
    ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
    ownership_neutral - ownership mask for neutral cells that are
    passable (1 if cell is neutral, 0 otherwise)
    """

    def __init__(self, grid: np.ndarray, _agents: list[str]):
        self._armies: np.ndarray = np.where(np.isin(grid, valid_generals), 1, 0).astype(int)
        self._generals: np.ndarray = np.where(np.isin(grid, valid_generals), 1, 0).astype(bool)
        self._mountains: np.ndarray = np.where(grid == MOUNTAIN, 1, 0).astype(bool)
        self._cities: np.ndarray = np.where(np.char.isdigit(grid), 1, 0).astype(bool)
        self._passable: np.ndarray = (grid != MOUNTAIN).astype(bool)

        self._ownership: dict[str, np.ndarray] = {
            "neutral": ((grid == PASSABLE) | (np.char.isdigit(grid))).astype(bool)
        }
        for i, agent in enumerate(_agents):
            self._ownership[agent] = np.where(grid == chr(ord("A") + i), 1, 0).astype(bool)

        # City costs are 40 + digit in the cell
        city_costs = np.where(np.char.isdigit(grid), grid, "0").astype(int)
        self.armies += 40 * self.cities + city_costs

    def get_visibility(self, agent_id: str) -> np.ndarray:
        channel = self._ownership[agent_id]
        return maximum_filter(channel, size=3)

    @staticmethod
    def channel_to_indices(channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.
        """
        return np.argwhere(channel != 0)

    @property
    def ownership(self) -> dict[str, np.ndarray]:
        return self._ownership

    @ownership.setter
    def ownership(self, value):
        self._ownership = value

    @property
    def armies(self) -> np.ndarray:
        return self._armies

    @armies.setter
    def armies(self, value):
        self._armies = value

    @property
    def generals(self) -> np.ndarray:
        return self._generals

    @generals.setter
    def generals(self, value):
        self._generals = value

    @property
    def mountains(self) -> np.ndarray:
        return self._mountains

    @mountains.setter
    def mountains(self, value):
        self._mountains = value

    @property
    def cities(self) -> np.ndarray:
        return self._cities

    @cities.setter
    def cities(self, value):
        self._cities = value

    @property
    def passable(self) -> np.ndarray:
        return self._passable

    @passable.setter
    def passable(self, value):
        self._passable = value

    @property
    def ownership_neutral(self) -> np.ndarray:
        return self._ownership["neutral"]

    @ownership_neutral.setter
    def ownership_neutral(self, value):
        self._ownership["neutral"] = value
