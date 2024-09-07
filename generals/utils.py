import time
import numpy as np
from importlib.resources import files
from .constants import PASSABLE, MOUNTAIN
from copy import deepcopy
from generals.env import GameConfig

from typing import List, Dict, Tuple


def map_from_generator(
    grid_size: int = 10,
    mountain_density: float = 0.2,
    town_density: float = 0.05,
    general_positions: List[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Generate a map with the given parameters.

    Args:
        grid_size: int, size of the grid
        mountain_density: float, probability of mountain in a cell
        town_density: float, probability of town in a cell
        general_positions: List[Tuple[int, int]], positions of generals
    """

    spatial_dim = (grid_size, grid_size)
    p_neutral = 1 - mountain_density - town_density
    probs = [p_neutral, mountain_density] + [town_density / 10] * 10

    # Place parts of the map
    map = np.random.choice(
        [PASSABLE, MOUNTAIN, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], size=spatial_dim, p=probs
    )

    # place generals on random squares
    if general_positions is None:
        general_positions = np.random.choice(
            grid_size, size=(2, 2), replace=False
        )

    for i, idx in enumerate(general_positions):
        map[idx[0], idx[1]] = chr(ord('A') + i)

    # generate until valid map
    return (
        map
        if validate_map(map)
        else map_from_generator(grid_size, mountain_density, town_density)
    )


def map_from_string(map_str: str) -> np.ndarray:
    """
    Convert map from string to np.ndarray.

    Args:
        map_str: str, map layout as string
    """
    map_list = map_str.strip().split("\n")
    map = np.array([list(row) for row in map_list])
    return map


def map_from_file(map_name: str) -> np.ndarray:
    """
    Load map from file.

    Args:
        map_name: str, name of the map file

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
    game_config = GameConfig(map=map_string)
    env = generals_v0(game_config)
    map = map_from_string(map_string)
    _ = env.reset(map, seed=42)
    game_states = [deepcopy(env.game.channels)]
    for i in range(len(action_sequence)):
        actions = {}
        for agent in env.agents:
            actions[agent] = action_sequence[i][agent]
        _ = env.step(actions)
        game_states.append(deepcopy(env.game.channels))

    return map, game_states

class Player:
    def __init__(self, name):
        self.name = name

    def play(self, observation):
        mask = observation['action_mask']
        valid_actions = np.argwhere(mask == 1)
        action = np.random.choice(len(valid_actions))
        return valid_actions[action]

    def __str__(self):
        return self.name

def run(game_config, agents: List[Player] = []):
    from generals.env import generals_v0
    if game_config.replay_file is not None:
        map, game_states = load_replay(game_config.replay_file)
    else:
        if game_config.map is not None:
            map = map_from_string(game_config.map)
        else:
            map = map_from_generator(
                grid_size=game_config.grid_size,
                mountain_density=game_config.mountain_density,
                town_density=game_config.town_density,
                general_positions=game_config.general_positions,
            )
        env = generals_v0(game_config)
        _ = env.reset(map)
        game_states = [deepcopy(env.game.channels)]

    assert validate_map(map), "Map is invalid"
    assert len(agents) == 2, "Exactly two agents must be provided"
    env = generals_v0(map)
    agents = {agent.name: agent for agent in agents}

    env.reset(map)
    env.render()

    game_step, last_input_time, last_move_time = 0, 0, 0

    while 1:
        _t = time.time()
        if _t - last_input_time > 0.008: # check for input every 8ms
            control_events = env.renderer.handle_events()
            last_input_time = _t
            env.render()
        else:
            control_events = {"time_change": 0}
        
        # if we control replay, change game state
        game_step = max(0, min(len(game_states) - 1, game_step + control_events["time_change"]))
        if env.renderer.paused and game_step != env.game.time:
            env.game.channels = deepcopy(game_states[game_step])
            env.game.time = game_step
            last_move_time = _t
        # if we are not paused, play the game
        elif _t - last_move_time > env.renderer.game_speed * 0.512 and not env.renderer.paused:
            observations = env.game.get_all_observations()
            actions = {}
            for agent in env.agents:
                actions[agent] = agents[agent].play(observations[agent])
            _ = env.step(actions)
            game_step = env.game.time
            game_states = game_states[:game_step]
            game_states.append(deepcopy(env.game.channels))
            last_move_time = _t
