from collections import deque
from typing import Literal

import numpy as np

from .config import MOUNTAIN, PASSABLE

DEFAULT_MIN_GRID_DIM = (18, 18)
DEFAULT_MAX_GRID_DIM = (23, 23)
DEFAULT_MOUNTAIN_DENSITY = 0.20
DEFAULT_CITY_DENSITY = 0.05
MAX_GENERALSIO_ATTEMPTS = 20
MIN_GENERALS_DISTANCE = 20
RADIUS_FROM_GENERAL = [6]


class InvalidGridError(Exception):
    pass


class Grid:
    """
    Represents the game grid containing passable areas, mountains, cities, and generals.

    Attributes:
        grid (np.ndarray): 2D array representing the grid.
    """

    def __init__(self, grid: str | np.ndarray):
        """
        Initializes the Grid either from a string or a NumPy array.

        Args:
            grid (str | np.ndarray): The grid representation.

        Raises:
            ValueError: If grid is not a string or NumPy array.
            InvalidGridError: If the grid layout is invalid.
        """
        if not isinstance(grid, str | np.ndarray):
            raise ValueError(f"grid must be a str or np.ndarray. Received grid with type: {type(grid)}.")

        if isinstance(grid, str):
            grid = grid.strip()
            grid = Grid.numpify_grid(grid)

        Grid.ensure_grid_is_valid(grid)
        self.grid = grid

    @property
    def shape(self):
        return self.grid.shape

    @staticmethod
    def ensure_grid_is_valid(grid: np.ndarray):
        if not Grid.are_generals_connected(grid):
            raise InvalidGridError("Invalid grid layout - generals cannot reach each other.")

        # check that exactly one 'A' and one 'B' are present in the grid
        first_general = np.argwhere(np.isin(grid, ["A"]))
        second_general = np.argwhere(np.isin(grid, ["B"]))
        if len(first_general) != 1 or len(second_general) != 1:
            raise InvalidGridError("Exactly one 'A' and one 'B' should be present in the grid.")

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

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
    def are_generals_connected(grid: np.ndarray | str) -> bool:
        """
        Returns True if there is a path connecting the two generals.
        """
        if isinstance(grid, str):
            grid = Grid.numpify_grid(grid)

        height, width = grid.shape

        def dfs(grid, visited, square):
            i, j = square
            if i < 0 or i >= height or j < 0 or j >= width or visited[i, j]:
                return
            if grid[i, j] == MOUNTAIN or str(grid[i, j]).isdigit() or grid[i, j] == "x":  # mountain or city
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
        return Grid.stringify_grid(self.grid)


class GridFactory:
    def __init__(
        self,
        mode: Literal["uniform", "generalsio"] = "uniform",
        min_grid_dims: tuple[int, int] = DEFAULT_MIN_GRID_DIM,
        max_grid_dims: tuple[int, int] = DEFAULT_MAX_GRID_DIM,
        mountain_density: float = DEFAULT_MOUNTAIN_DENSITY,
        city_density: float = DEFAULT_CITY_DENSITY,
        min_generals_distance: int = MIN_GENERALS_DISTANCE,
        general_positions: list[tuple[int, int]] | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            mode: Either "uniform" or "generalsio". If "uniform", the grid will be generated uniformly randomly,
            based on specified probabilities. If "generalsio", the grid will be generated to be similar to generalsio.
            min_grid_dims: The minimum (inclusive) height & width of the grid.
            max_grid_dims: The maximum (inclusive) height & width of the grid.
            mountain_density: The probability any given square is a mountain.
            city_density: The probability any given square is a city.
            general_positions: The (row, col) of each general.
            seed: A random seed i.e. a way to make the randomness repeatable.
        """
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.min_grid_dims = min_grid_dims
        self.max_grid_dims = max_grid_dims
        self.mountain_density = mountain_density
        self.city_density = city_density
        self.general_positions = general_positions
        self.min_generals_distance = min_generals_distance
        assert self.mode in ["uniform", "generalsio"], f"Invalid mode: {self.mode}"

    def set_rng(self, rng: np.random.Generator):
        self.rng = rng

    def generate(self) -> Grid:
        if self.mode == "uniform":
            return self.generate_uniform_grid()
        elif self.mode == "generalsio":
            return self.generate_generalsio_grid()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def generate_generalsio_grid(self) -> Grid:
        grid_height = self.rng.integers(DEFAULT_MIN_GRID_DIM[0], DEFAULT_MAX_GRID_DIM[0] + 1)
        grid_width = self.rng.integers(DEFAULT_MIN_GRID_DIM[1], DEFAULT_MAX_GRID_DIM[1] + 1)
        grid_dims = (grid_height, grid_width)
        num_tiles = grid_height * grid_width

        # Counts based on real generals.io 1v1 queue
        cities_to_place = 5 + self.rng.choice([4, 5, 6])
        num_mountains = int(DEFAULT_MOUNTAIN_DENSITY * num_tiles + 0.02 * num_tiles * self.rng.random())

        def bfs_distance(start, grid):
            distances = np.full(grid_dims, float("inf"))
            distances[start] = 0
            queue = deque([start])

            # Possible moves: up, right, down, left
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

            while queue:
                current = queue.popleft()
                current_dist = distances[current]

                for move in moves:
                    next_pos = (current[0] + move[0], current[1] + move[1])

                    # Check bounds
                    if 0 <= next_pos[0] < grid_dims[0] and 0 <= next_pos[1] < grid_dims[1]:
                        # Check if passable (not a mountain) and not visited
                        if grid[next_pos] != MOUNTAIN and distances[next_pos] == float("inf"):
                            distances[next_pos] = current_dist + 1
                            queue.append(next_pos)

            return distances

        def create_distance_mask(distances, max_distance):
            return (distances <= max_distance) & (distances > 0)

        # Initialize empty map
        map = np.full(grid_dims, PASSABLE, dtype=str)

        # Place mountains randomly
        self._place_mountains(map, num_mountains)

        g1 = (self.rng.integers(grid_height), self.rng.integers(grid_width))
        distances_from_g1 = bfs_distance(g1, map)

        max_attempts = MAX_GENERALSIO_ATTEMPTS // 5
        g2 = None
        for _ in range(max_attempts):
            candidate_g2 = (self.rng.integers(grid_height), self.rng.integers(grid_width))
            if distances_from_g1[candidate_g2] >= self.min_generals_distance and distances_from_g1[
                candidate_g2
            ] != float("inf"):
                g2 = candidate_g2
                break

        if g2 is None:
            # If we couldn't place g2 after max attempts, start over with a new grid
            return self.generate_generalsio_grid()

        general_positions = [g1, g2]

        distances_from_g2 = bfs_distance(g2, map)

        # Place one city close to each general
        for distance in RADIUS_FROM_GENERAL:
            mask_close_g1 = create_distance_mask(distances_from_g1, distance)
            mask_close_g2 = create_distance_mask(distances_from_g2, distance)
            for mask in [mask_close_g1, mask_close_g2]:
                valid_positions = np.where(mask)

                # Randomly select one position from the valid positions
                idx = self.rng.integers(0, len(valid_positions[0]))
                city_pos = (valid_positions[0][idx], valid_positions[1][idx])
                # Generate city value (0 - 9 or 'x')
                city_cost = self.rng.choice([str(i) for i in range(10)] + ["x"])
                map[city_pos] = city_cost

                cities_to_place -= 1

        # Place remaining cities
        self._place_cities(map, cities_to_place)

        # Calculate number of passable cells in radius 5 of both generals
        radius = 5

        # Use the distance maps we already calculated
        mask_radius_g1 = create_distance_mask(distances_from_g1, radius)
        mask_radius_g2 = create_distance_mask(distances_from_g2, radius)

        # Count passable cells (not mountains or cities) within radius
        passable_mask = map == PASSABLE
        passable_cells_g1 = np.sum(mask_radius_g1 & passable_mask)
        passable_cells_g2 = np.sum(mask_radius_g2 & passable_mask)

        if abs(passable_cells_g1 - passable_cells_g2) > 10:
            return self.generate_generalsio_grid()

        for i, idx in enumerate(general_positions):
            map[idx[0], idx[1]] = chr(ord("A") + i)

        # Convert map to string
        map_string = "\n".join(["".join(row) for row in map.astype(str)])

        try:
            return Grid(map_string)
        except InvalidGridError:
            # Keep randomly generating grids until one works!
            return self.generate_generalsio_grid()

    def generate_uniform_grid(self) -> Grid:
        grid_height = self.rng.integers(self.min_grid_dims[0], self.max_grid_dims[0] + 1)
        grid_width = self.rng.integers(self.min_grid_dims[0], self.max_grid_dims[0] + 1)
        grid_dims = (grid_height, grid_width)

        # Probabilities of each cell type
        p_neutral = 1 - self.mountain_density - self.city_density
        if p_neutral < 0:
            raise ValueError("Sum of mountain_density and city_density cannot exceed 1.")
        # Distribute city_density across city types
        p_cities = self.city_density / 11  # 10 digits + 'x'
        probs = [p_neutral, self.mountain_density] + [p_cities] * 10 + [p_cities]  # For "x"

        # Place cells on the map
        map = self.rng.choice(
            [PASSABLE, MOUNTAIN, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "x"],
            size=grid_dims,
            p=probs,
        )

        general_positions = self.general_positions
        if general_positions is None:
            # Select each generals location, they should be atleast some distance apart
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
        except InvalidGridError:
            # Keep randomly generating grids until one works!
            return self.generate_uniform_grid()

    def _place_mountains(self, map: np.ndarray, num_mountains: int):
        available_positions = np.argwhere(map == PASSABLE)
        selected_indices = self.rng.choice(len(available_positions), size=num_mountains, replace=False)
        selected_positions = available_positions[selected_indices]
        map[selected_positions[:, 0], selected_positions[:, 1]] = MOUNTAIN

    def _place_cities(self, map: np.ndarray, cities_to_place: int):
        mountain_positions = np.argwhere(map == MOUNTAIN)
        selected_indices = self.rng.choice(len(mountain_positions), size=cities_to_place, replace=False)
        selected_positions = mountain_positions[selected_indices]
        city_costs = self.rng.choice([str(i) for i in range(10)] + ["x"], size=cities_to_place)
        map[selected_positions[:, 0], selected_positions[:, 1]] = city_costs
