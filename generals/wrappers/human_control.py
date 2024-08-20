import time
import numpy as np
from copy import deepcopy
from pettingzoo.utils.wrappers.base import BaseWrapper
from generals.env import Generals, generals_v0
import generals.utils

def run(map: np.ndarray, replay: str = None):
    if replay is not None:
        map, action_sequence = generals.utils.load_replay(replay)
    env = generals_v0(map) # created map

    # Load frames from replays
    index = 0
    o, i = env.reset(seed=42, options={"replay": "test"})
    game_states = [deepcopy(env.game.channels)]
    while env.agents:
        actions = {}
        for agent in env.agents:
            actions[agent] = action_sequence[index][agent]
        o, r, te, tr, i = env.step(actions)
        game_states.append(deepcopy(env.game.channels))
        index += 1
    # Give actions sequences to agents


    # Game loop where we send timestamps to agents

    ###
    env.game.channels = game_states[0]
    while True:
        env.render()
        time.sleep(0.064)



map = generals.utils.generate_map(
    grid_size=16,
    mountain_density=0.2,
    town_density=0.05,
    n_generals=2,
    general_positions=None,
)

run(map, "test")
