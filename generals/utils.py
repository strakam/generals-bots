import numpy as np
from importlib.resources import files
from .constants import PASSABLE, MOUNTAIN, CITY, GENERAL
from copy import deepcopy

from typing import List, Dict, Tuple


def map_from_generator(
    grid_size: int = 10,
    mountain_density: float = 0.2,
    town_density: float = 0.05,
    n_generals: int = 2,
    general_positions: List[Tuple[int, int]] = None,
) -> np.ndarray:
    spatial_dim = (grid_size, grid_size)
    p_neutral = 1 - mountain_density - town_density
    probs = [p_neutral, mountain_density] + [town_density / 10] * 10
    map = np.random.choice(
        [PASSABLE, MOUNTAIN, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=spatial_dim, p=probs
    )
    # print(mountain_density, town_density, n_generals)
    # print(f'Grid size is {grid_size**2}')
    # print(f'Number of towns is {np.sum(map == CITY)} which is {np.sum(map == CITY) / grid_size**2 * 100}%')
    # print(f'Number of mountains is {np.sum(map == MOUNTAIN)} which is {np.sum(map == MOUNTAIN) / grid_size**2 * 100}%')


    # place generals on random squares
    if general_positions is None:
        general_positions = np.random.choice(
            grid_size, size=(n_generals, 2), replace=False
        )

    for i, idx in enumerate(general_positions):
        # convert 'A'
        map[idx[0], idx[1]] = chr(ord('A') + i)

    # generate until valid map
    return (
        map
        if validate_map(map)
        else map_from_generator(grid_size, mountain_density, town_density, n_generals)
    )


def map_from_string(map_str: str) -> np.ndarray:
    """
    Convert map from string to np.ndarray.

    Args:
        map_str: str
    """
    map_list = map_str.strip().split("\n")
    map = np.array([list(row) for row in map_list])
    return map


def map_from_file(map_name: str) -> np.ndarray:
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
            map = np.array([list(line.strip()) for line in f])
            validity = validate_map(map)
            if not validity:
                raise ValueError(
                    "The map is invalid, because generals are separated by mountains"
                )
            return map
    except ValueError:
        raise ValueError("Invalid map format or shape")


def validate_map(map: str) -> bool:
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

    generals = np.argwhere(np.isin(map, ['A', 'B']))  # hardcoded for now
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
        map = "\n".join(["".join([cell for cell in row]) for row in map])
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
    # Load map and actions
    with open(path, "r") as f:
        lines = f.readlines()
        # take rows until first empty line
        rows = lines[: lines.index("\n")]
        map_string = "".join(rows)
        map = map_from_string(map_string)
        # after empty line, read actions
        action_sequence = []
        for line in lines[lines.index("\n") + 1 :]:
            action_sequence.append(
                {
                    player.split(":")[0]: np.array(
                        player.split(":")[1].split(" "), dtype=np.int32
                    )
                    for player in line.strip().split(",")
                }
            )
    # Play actions to recreate states that happened
    from generals.env import generals_v0
    env = generals_v0(map)
    _, _ = env.reset(seed=42)
    game_states = [deepcopy(env.game.channels)]
    for i in range(len(action_sequence)):
        actions = {}
        for agent in env.agents:
            actions[agent] = action_sequence[i][agent]
        _ = env.step(actions)
        game_states.append(deepcopy(env.game.channels))

    return map, game_states
