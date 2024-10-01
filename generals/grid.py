import numpy as np
from generals.config import PASSABLE, MOUNTAIN


class Grid:
    def __init__(self, grid: str):
        if not Grid.verify_map(grid):
            raise ValueError("Invalid map layout - generals cannot reach each other.")
        self._grid_string = grid.strip()

    @property
    def grid(self):
        return self._grid_string

    @grid.setter
    def grid(self, grid: str):
        grid = grid.strip()
        if not Grid.verify_map(grid):
            raise ValueError("Invalid map layout - generals cannot reach each other.")
        self._grid_string = grid

    @property
    def numpified_grid(self):
        return Grid.numpify_grid(self._grid_string)

    @staticmethod
    def numpify_grid(grid: str) -> np.ndarray:
        return np.array([list(row) for row in grid.strip().split("\n")])

    @staticmethod
    def stringify_grid(grid: np.ndarray) -> str:
        return "\n".join(["".join(row) for row in grid])

    @staticmethod
    def verify_map(map: str) -> bool:
        """
        Verify map layout (can generals reach each other?)
        Returns True if map is valid, False otherwise
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

        map = Grid.numpify_grid(map)
        generals = np.argwhere(np.isin(map, ["A", "B"]))
        start, end = generals[0], generals[1]
        visited = np.zeros_like(map, dtype=bool)
        dfs(map, visited, start)
        return visited[end[0], end[1]]

    def __str__(self):
        return self._grid_string


class GridFactory:
    def __init__(
        self,
        grid_dims: tuple[int, int] = (10, 10),
        mountain_density: float = 0.2,
        city_density: float = 0.05,
        general_positions: list[tuple[int, int]] = None,
        seed: int = None,
    ):
        self.grid_height = grid_dims[0]
        self.grid_width = grid_dims[1]
        self.mountain_density = mountain_density
        self.city_density = city_density
        self.general_positions = general_positions
        self.seed = seed

    def grid_from_string(self, grid: str) -> Grid:
        return Grid(grid)

    def grid_from_generator(
        self,
        grid_dims: tuple[int, int] = None,
        mountain_density: float = None,
        city_density: float = None,
        general_positions: list[tuple[int, int]] = None,
        seed: int = None,
    ) -> Grid:
        if grid_dims is None:
            grid_dims = (self.grid_height, self.grid_width)
        if mountain_density is None:
            mountain_density = self.mountain_density
        if city_density is None:
            city_density = self.city_density
        if general_positions is None:
            general_positions = self.general_positions
        if seed is None:
            seed = self.seed

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

        # Convert map to string
        map_string = "\n".join(["".join(row) for row in map.astype(str)])

        try:
            return Grid(map_string)
        except ValueError:
            return self.grid_from_generator(
                grid_dims=grid_dims,
                mountain_density=mountain_density,
                city_density=city_density,
                general_positions=general_positions,
                seed=seed,
            )