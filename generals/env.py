import functools

import gymnasium
from gymnasium.spaces import Discrete

import pettingzoo
from pettingzoo.utils import wrappers



def generals_v0(render_mode=None):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(num_players=2, map_size=10, max_turns=100, observation_type="numpy")
    # Apply parallel_to_aec to support AEC api
    env = pettingzoo.utils.parallel_to_aec(env)
    return env


class Generals(pettingzoo.ParallelEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_players, map_size, max_turns, observation_type, debug=False):
        print('inittin')
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


