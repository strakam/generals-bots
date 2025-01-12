import dataclasses

import numpy as np


@dataclasses.dataclass
class Observation(dict):
    """
    Hybrid implementation that maintains dict-like behavior for compatibility
    while offering memory-efficient tensor operations.
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
    pad_to: int | None = None

    def __post_init__(self):
        if self.pad_to is not None:
            self.apply_padding()

    def __getitem__(self, attribute_name: str):
        return getattr(self, attribute_name)

    def keys(self):
        return dataclasses.asdict(self).keys()

    def values(self):
        return dataclasses.asdict(self).values()

    def items(self):
        return dataclasses.asdict(self).items()

    def apply_padding(self):
        h_pad = (0, self.pad_to - self.armies.shape[0])
        w_pad = (0, self.pad_to - self.armies.shape[1])

        for field in dataclasses.fields(self):
            if isinstance(getattr(self, field.name), np.ndarray):
                value = 1 if field.name == "mountains" else 0
                setattr(
                    self,
                    field.name,
                    np.pad(getattr(self, field.name), (h_pad, w_pad), "constant", constant_values=value),
                )

    def as_tensor(self) -> np.ndarray:
        """
        Returns a 3D tensor of shape (15, rows, cols). Suitable for neural nets.
        """
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
                np.full_like(self.armies, self.owned_land_count),
                np.full_like(self.armies, self.owned_army_count),
                np.full_like(self.armies, self.opponent_land_count),
                np.full_like(self.armies, self.opponent_army_count),
                np.full_like(self.armies, self.timestep),
                np.full_like(self.armies, self.priority),
            ],
            axis=0,
        )
