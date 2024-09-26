import time
import numpy as np
from generals.config import GameConfig
from generals.agents import RandomAgent
from copy import deepcopy

from typing import List, Dict



def store_replay(
    map: np.ndarray,
    action_sequence: List[Dict[str, np.ndarray]],
    name: str = None,
):
    print(f"Storing replay {name}")
    with open(name, "w") as f:
        players = list(action_sequence[0].keys())
        f.write(f"{players[0]} vs {players[1]}\n")
        map = "\n".join(["".join([cell for cell in row]) for row in map])
        f.write(f"{map}\n\n")
        for player_actions in action_sequence:
            player_string = {}
            for player in players:
                act = player_actions[player]
                player_string[player] = (
                    f"{player}:{act[0]} {act[1]} {act[2]} {act[3]} {act[4]}"
                )
            # join all player strings with comma and write to file
            f.write(",".join(player_string.values()) + "\n")


def load_replay(path: str):
    print(f"Loading replay {path}")
    # Load map and actions
    with open(path, "r") as f:
        lines = f.readlines()
        players = lines[0].strip().split(" vs ")

        # take rows until first empty line
        rows = lines[1 : lines.index("\n")]
        map = "".join(rows)

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
    from generals.env import pz_generals
    agents = [RandomAgent(name=players[0]), RandomAgent(name=players[1])]
    game_config = GameConfig(
        agents=agents,
    )
    env = pz_generals(game_config)
    _ = env.reset(map, seed=42)

    game_states = [deepcopy(env.game.channels)]
    for i in range(len(action_sequence)):
        actions = {}
        for agent in env.agents:
            actions[agent] = action_sequence[i][agent]
        _ = env.step(actions)
        game_states.append(deepcopy(env.game.channels))

    return players, map, game_states


def run_replay(replay_file: str):
    agents, map, game_states = load_replay(replay_file)
    from generals.env import pz_generals
    agents = [RandomAgent(name=agents[0],color=(67,99,216)), RandomAgent(name=agents[1])]
    game_config = GameConfig(
        agents=agents,
    )
    env = pz_generals(game_config, render_mode="human")
    env.reset(map, options={"from_replay": True})
    env.renderer.render()

    game_step, last_input_time, last_move_time = 0, 0, 0
    while 1:
        _t = time.time()
        # Check inputs
        if _t - last_input_time > 0.008:  # check for input every 8ms
            control_events = env.renderer.render()
            last_input_time = _t
        else:
            control_events = {"time_change": 0}
        # If we control replay, change game state
        game_step = max(
            0, min(len(game_states) - 1, game_step + control_events["time_change"])
        )
        if env.renderer.paused and game_step != env.game.time:
            env.agents = deepcopy(env.possible_agents)
            env.game.channels = deepcopy(game_states[game_step])
            env.game.time = game_step
            last_move_time = _t
        # If we are not paused, play the game
        elif (
            _t - last_move_time > env.renderer.game_speed * 0.512
            and not env.renderer.paused
        ):
            if env.game.is_done():
                env.renderer.paused = True
            game_step = min(len(game_states) - 1, game_step + 1)
            env.game.channels = deepcopy(game_states[game_step])
            env.game.time = game_step
            last_move_time = _t
        env.renderer.clock.tick(60)
