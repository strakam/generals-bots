import numpy as np
from typing import List, Tuple
from generals.config import PASSABLE, MOUNTAIN


class Mapper:
    def __init__(
        self,
        grid_dims: Tuple[int, int] = (10, 10),
        mountain_density: float = 0.2,
        city_density: float = 0.05,
        general_positions: List[Tuple[int, int]] = None,
        seed: int = None,
    ):
        self.grid_height = grid_dims[0]
        self.grid_width = grid_dims[1]
        self.mountain_density = mountain_density
        self.city_density = city_density
        self.general_positions = general_positions
        self.seed = seed

        self.map = self.generate_map(
            grid_dims=grid_dims,
            mountain_density=mountain_density,
            city_density=city_density,
            general_positions=general_positions,
            seed=seed,
        )

    def get_map(self, numpify=False) -> np.ndarray:
        if numpify:  # Return map as np.ndarray
            return self.numpify_map(self.map)
        return self.map  # Return map as string

    def reset(self):
        self.map = self.generate_map(
            grid_dims=(self.grid_height, self.grid_width),
            mountain_density=self.mountain_density,
            city_density=self.city_density,
            general_positions=self.general_positions,
            seed=self.seed,
        )

    @staticmethod
    def generate_map(
        grid_dims: Tuple[int, int] = (10, 10),
        mountain_density: float = 0.2,
        city_density: float = 0.05,
        general_positions: List[Tuple[int, int]] = None,
        seed: int = None,
    ) -> np.ndarray:

        # Probabilities of each cell type
        p_neutral = 1 - mountain_density - city_density
        probs = [p_neutral, mountain_density] + [city_density / 10] * 10

        # Place cells on the map
        rng = np.random.default_rng(seed)
        map = rng.choice(
            [PASSABLE, MOUNTAIN, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            size=grid_dims,
            p=probs,
        )

        # Place generals on random squares - generals_positions is a list of two tuples
        if general_positions is None:
            general_positions = []
            while len(general_positions) < 2:
                position = tuple(rng.integers(0, grid_dims))
                if position not in general_positions:
                    general_positions.append(position)

        for i, idx in enumerate(general_positions):
            map[idx[0], idx[1]] = chr(ord("A") + i)

        map = Mapper.stringify_map(map)

        # Iterate until map is valid
        if Mapper.validate_map(map):
            return map
        else:
            return Mapper.generate_map(
                grid_dims=grid_dims,
                mountain_density=mountain_density,
                city_density=city_density,
                general_positions=general_positions,
                seed=seed,
            )

    @staticmethod
    def validate_map(map: str) -> bool:
        """
        Validate map layout.
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

        map = Mapper.numpify_map(map)
        generals = np.argwhere(np.isin(map, ["A", "B"]))
        start, end = generals[0], generals[1]
        visited = np.zeros_like(map, dtype=bool)
        dfs(map, visited, start)
        return visited[end[0], end[1]]

    @staticmethod
    def numpify_map(map: str) -> np.ndarray:
        return np.array([list(row) for row in map.strip().split("\n")])

    @staticmethod
    def stringify_map(map: np.ndarray) -> str:
        return "\n".join(["".join(row) for row in map])
