import functools

import gymnasium
from gymnasium.spaces import Discrete

import pettingzoo
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers



def env(render_mode=None):
    pass

def raw_env(render_mode=None):
    pass


class Generals(ParallelEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_players, map_size, max_turns, observation_type, debug=False):
        pass

    def render(self, mode='human'):
        pass

    def reset(self):
        pass

    def step(self, actions):
        pass

    def observe(self, agent):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
