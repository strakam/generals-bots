import numpy as np
from numpy.random import Generator

from .config import MOUNTAIN, PASSABLE


class Grid:
    def __init__(self, grid: str | np.ndarray):
        self.grid = grid

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid: str | np.ndarray):
        match grid:
            case str(grid):
                grid = grid.strip()
                grid = Grid.numpify_grid(grid)
            case np.ndarray():
                pass
            case _:
                raise ValueError("Grid must be encoded as a string or a numpy array.")
        if not Grid.verify_grid_connectivity(grid):
            raise ValueError("Invalid grid layout - generals cannot reach each other.")
        # check that exactly one 'A' and one 'B' are present in the grid
        first_general = np.argwhere(np.isin(grid, ["A"]))
        second_general = np.argwhere(np.isin(grid, ["B"]))
        if len(first_general) != 1 or len(second_general) != 1:
            raise ValueError("Exactly one 'A' and one 'B' should be present in the grid.")

        self._grid = grid

    @staticmethod
    def generals_distance(grid: "Grid") -> int:
        generals = np.argwhere(np.isin(grid.grid, ["A", "B"]))
        return abs(generals[0][0] - generals[1][0]) + abs(generals[0][1] - generals[1][1])

    @staticmethod
    def numpify_grid(grid: str) -> np.ndarray:
        return np.array([list(row) for row in grid.strip().split("\n")])

    @staticmethod
    def stringify_grid(grid: np.ndarray) -> str:
        return "\n".join(["".join(row) for row in grid])

    @staticmethod
    def verify_grid_connectivity(grid: np.ndarray | str) -> bool:
        """
        Verify grid layout (can generals reach each other?)
        Returns True if grid is valid, False otherwise
        """
        if isinstance(grid, str):
            grid = Grid.numpify_grid(grid)

        height, width = grid.shape

        def dfs(grid, visited, square):
            i, j = square
            if i < 0 or i >= height or j < 0 or j >= width or visited[i, j]:
                return
            if grid[i, j] == MOUNTAIN or str(grid[i, j]).isdigit():  # mountain or city
                return
            visited[i, j] = True
            for di, dj in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                new_square = (i + di, j + dj)
                dfs(grid, visited, new_square)

        generals = np.argwhere(np.isin(grid, ["A", "B"]))
        start, end = generals[0], generals[1]

        visited = np.zeros_like(grid, dtype=bool)
        dfs(grid, visited, start)
        return visited[end[0], end[1]]

    def __str__(self):
        return Grid.stringify_grid(self._grid)


class GridFactory:
    def __init__(
        self,
        grid_dims: tuple[int, int] = (10, 10),
        mountain_density: float = 0.2,
        city_density: float = 0.05,
        general_positions: list[tuple[int, int]] | None = None,
        seed: int | None = None,
    ):
        self.grid_height = grid_dims[0]
        self.grid_width = grid_dims[1]
        self.mountain_density = mountain_density
        self.city_density = city_density
        self.general_positions = general_positions
        self._rng = np.random.default_rng(seed)

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, number_generator: Generator):
        self._rng = number_generator

    def grid_from_string(self, grid: str) -> Grid:
        return Grid(grid)

    def grid_from_generator(
        self,
        grid_dims: tuple[int, int] | None = None,
        mountain_density: float | None = None,
        city_density: float | None = None,
        general_positions: list[tuple[int, int]] | None = None,
        seed: int | None = None,
    ) -> Grid:
        if grid_dims is None:
            grid_dims = (self.grid_height, self.grid_width)
        if mountain_density is None:
            mountain_density = self.mountain_density
        if city_density is None:
            city_density = self.city_density
        if general_positions is None:
            general_positions = self.general_positions
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Probabilities of each cell type
        p_neutral = 1 - mountain_density - city_density
        probs = [p_neutral, mountain_density] + [city_density / 10] * 10

        # Place cells on the map
        map = self.rng.choice(
            [PASSABLE, MOUNTAIN, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            size=grid_dims,
            p=probs,
        )

        # Place generals on random squares, they should be atleast some distance apart
        min_distance = max(grid_dims) // 2
        p1 = self.rng.integers(0, grid_dims[0]), self.rng.integers(0, grid_dims[1])
        while True:
            p2 = self.rng.integers(0, grid_dims[0]), self.rng.integers(0, grid_dims[1])
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) >= min_distance:
                break
        general_positions = [p1, p2]
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
                seed=None,
            )
