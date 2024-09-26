import numpy as np
from typing import List, Tuple
from generals.config import PASSABLE, MOUNTAIN


class Mapper:
    def __init__(
        self,
        grid_size: int = 10,
        mountain_density: float = 0.2,
        city_density: float = 0.05,
        general_positions: List[Tuple[int, int]] = None,
        seed: int = None,
    ):
        self.grid_size = grid_size
        self.mountain_density = mountain_density
        self.city_density = city_density
        self.general_positions = general_positions
        self.seed = seed

        self.map = self.generate_map()

    def generate_map(self) -> np.ndarray:
        spatial_dim = (self.grid_size, self.grid_size)

        # Probabilities of each cell type
        p_neutral = 1 - self.mountain_density - self.city_density
        probs = [p_neutral, self.mountain_density] + [self.city_density / 10] * 10

        # Place cells on the map
        rng = np.random.default_rng(self.seed)
        map = rng.choice(
            [PASSABLE, MOUNTAIN, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            size=spatial_dim,
            p=probs,
        )

        # Place generals on random squares
        if self.general_positions is None:
            self.general_positions = np.random.choice(
                self.grid_size, size=(2, 2), replace=False
            )

        for i, idx in enumerate(self.general_positions):
            map[idx[0], idx[1]] = chr(ord("A") + i)

        # iterate until map is valid
        return map if self.validate_map(map) else self.generate_map()

    def validate_map(self, map: str) -> bool:
        """
        Validate map layout.
        Args:
            map: np.ndarray

        Returns:
            bool: True if map is valid, False otherwise
        """

        def dfs(map, visited, square):
            i, j = square
            if (
                i < 0
                or i >= map.shape[0]
                or j < 0
                or j >= map.shape[1]
                or visited[i, j]
            ):
                return
            if map[i, j] == MOUNTAIN:
                return
            visited[i, j] = True
            for di, dj in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                new_square = (i + di, j + dj)
                dfs(map, visited, new_square)

        generals = np.argwhere(np.isin(map, ["A", "B"]))
        start, end = generals[0], generals[1]
        visited = np.zeros_like(map, dtype=bool)
        dfs(map, visited, start)
        return visited[end[0], end[1]]

    def set_map_from_string(self, map_str: str) -> np.ndarray:
        """
        Convert map from string to np.ndarray.
        """
        map_str = map_str.strip("\n")
        self.map = np.array([list(row) for row in map_str.split("\n")])

    def get_map(self) -> np.ndarray:
        return self.map
