import time
import numpy as np
from copy import deepcopy
from generals.env import Generals, generals_v0
import pygame
import generals.utils

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
    agents = {agent: Player(agent) for agent in env.agents}
    ###
    f = 32
    env.game.channels = game_states[f]
    env.game.time = f
    t = 0
    last_time = 0
    while True:
        control_events = env.renderer.handle_events(env.game)
        t = max(0, min(len(game_states) - 1, t + control_events["time_change"]))
        if time.time() - last_time > 0.064:
            if env.renderer.paused:
                env.game.channels = game_states[t]
                env.game.time = t
            else:
                o = env.game.get_all_observations()
                actions = {}
                for agent in env.agents:
                    actions[agent] = agents[agent].play(o[agent])
                o, r, te, tr, i = env.step(actions)
                t = env.game.time

            last_time = time.time()
            env.render()



map = generals.utils.generate_map(
    grid_size=16,
    mountain_density=0.2,
    town_density=0.05,
    n_generals=2,
    general_positions=None,
)

run(map, "test")
