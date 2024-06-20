import numpy as np
from importlib.resources import files
from .constants import PASSABLE, MOUNTAIN, CITY, GENERAL

from typing import List, Dict, Tuple


def generate_map(
    grid_size: int = 10,
    mountain_density: float = 0.1,
    town_density: float = 0.1,
    n_generals: int = 2,
    general_positions: List[Tuple[int, int]] = None,
) -> np.ndarray:
    spatial_dim = (grid_size, grid_size)
    p_neutral = 1 - mountain_density - town_density
    probs = [p_neutral, mountain_density, town_density]
    map = np.random.choice(
        [PASSABLE, MOUNTAIN, CITY], size=spatial_dim, p=probs
    ).astype(np.float32)

    # place generals on random squares
    if general_positions is None:
        general_positions = np.random.choice(
            grid_size, size=(n_generals, 2), replace=False
        )

    for i, idx in enumerate(general_positions):
        map[idx[0], idx[1]] = GENERAL + i

    # generate until valid map
    return (
        map
        if validate_map(map)
        else generate_map(grid_size, mountain_density, town_density, n_generals)
    )


def map_from_string(map_str: str) -> np.ndarray:
    """
    Convert map from string to np.ndarray.

    Args:
        map_str: str
    """
    map_list = map_str.strip().split("\n")
    map = np.array([list(row) for row in map_list]).astype(np.float32)
    validity = validate_map(map)
    if not validity:
        raise ValueError(
            "The map is invalid, because generals are separated by mountains"
        )
    return map


def load_map(map_name: str) -> np.ndarray:
    """
    Load map from file.

    Args:
        map_name: str

    Returns:
        np.ndarray: map layout
    """
    try:
        file_ref = str(files("generals.maps") / map_name)
        with open(file_ref, "r") as f:
            map = np.array([list(line.strip()) for line in f]).astype(np.float32)
            validity = validate_map(map)
            if not validity:
                raise ValueError(
                    "The map is invalid, because generals are separated by mountains"
                )
            return map
    except ValueError:
        raise ValueError("Invalid map format or shape")


def validate_map(map: np.ndarray) -> bool:
    """
    Validate map layout.
    WORKS ONLY FOR 2 GENERALS (for now)

    Args:
        map: np.ndarray

    Returns:
        bool: True if map is valid, False otherwise
    """

    def dfs(map, visited, square):
        i, j = square
        if i < 0 or i >= map.shape[0] or j < 0 or j >= map.shape[1] or visited[i, j]:
            return
        if map[i, j] == MOUNTAIN:
            return
        visited[i, j] = True
        for di, dj in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            new_square = (i + di, j + dj)
            dfs(map, visited, new_square)

    generals = np.argwhere(np.isin(map, [3, 4]))  # hardcoded for now
    start, end = generals[0], generals[1]
    visited = np.zeros_like(map, dtype=bool)
    dfs(map, visited, start)
    return visited[end[0], end[1]]


def store_replay(
    map: np.ndarray,
    action_sequence: List[Dict[str, np.ndarray]],
    name: str = None,
):
    print(f"Storing replay {name}")
    with open(name, "w") as f:
        map = "\n".join(["".join([str(int(cell)) for cell in row]) for row in map])
        f.write(f"{map}\n\n")
        for player_action in action_sequence:
            # action shouldnt be printed with brackets
            row = ",".join(
                [
                    f"{player}:{' '.join([str(cell) for cell in action])}"
                    for player, action in player_action.items()
                ]
            )
            f.write(f"{row}\n")


def load_replay(path: str):
    print(f"Loading replay {path}")
    with open(path, "r") as f:
        lines = f.readlines()
        # take rows until first empty line
        map = np.array(
            [[int(cell) for cell in row.strip()] for row in lines[: lines.index("\n")]],
            dtype=np.float32,
        )
        # after empty line, read actions
        actions = []
        for line in lines[lines.index("\n") + 1 :]:
            actions.append(
                {
                    player.split(":")[0]: np.array(
                        player.split(":")[1].split(" "), dtype=np.int32
                    )
                    for player in line.strip().split(",")
                }
            )

    return map, actions
