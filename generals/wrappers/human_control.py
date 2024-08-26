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
        # check if there are any valid actions
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
    env = generals_v0(map) # created map
    env.reset()
    agents = {agent: Player(agent) for agent in env.agents}
    ###
    t = 0
    env.game.channels = game_states[t]
    env.game.time = t
    env.render()
    last_time = 0
    while 1:
        _t = time.time()
        control_events = env.renderer.handle_events(env.game)
        t = max(0, min(len(game_states) - 1, t + control_events["time_change"]))
        if env.renderer.paused and env.game.time != t:
            env.game.channels = game_states[t]
            env.game.time = t
            env.render()
            last_time = _t
        elif _t - last_time > env.renderer.game_speed * 0.512 and not env.renderer.paused:
            o = env.game.get_all_observations()
            actions = {}
            for agent in env.agents:
                actions[agent] = agents[agent].play(o[agent])
            o, r, te, tr, i = env.step(actions)
            t = env.game.time
            game_states.append(deepcopy(env.game.channels))
            # remove all elements from game_states after t
            game_states = game_states[: t + 1]
            env.render()
            last_time = _t
        elif "changed" in control_events:
            env.render()
            


map = generals.utils.generate_map(
    grid_size=16,
    mountain_density=0.2,
    town_density=0.05,
    n_generals=2,
    general_positions=None,
)

run(map, "test")
