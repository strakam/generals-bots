import functools
from copy import copy

import gymnasium
from gymnasium.spaces import Discrete

import pettingzoo
from pettingzoo.utils import wrappers

from . import game, game_config, utils




def generals_v0(game_config=game_config.GameConfig, render_mode="human"):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(game_config)
    # Apply parallel_to_aec to support AEC api
    env = pettingzoo.utils.parallel_to_aec(env)
    return env


class Generals(pettingzoo.ParallelEnv):
    metadata = {'render.modes': ["human", "none"]}

    def __init__(self, game_config: game_config.GameConfig, render_mode="human"):
        self.game_config = game_config
        self.render_mode = render_mode
        self.game = game.Game(game_config)
        self.visualizer = utils.Visualizer(game_config, self.game.grid)
        self.possible_agents = ["a", "b"]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)


    def render(self):
        if self.render_mode == "human":
            grid = self.game.grid
            self.visualizer.draw_grid(grid)

    def reset(self, seed=None, options=None):
        self.game = game.Game(self.game_config)
        self.agents = copy(self.possible_agents)

        observations = {
            a : 'kek' for a in self.agents
        }

        infos = {
            a : 'infou' for a in self.agents
        }

        return observations, infos


    def step(self, actions):
        # return some dummy values for now
        self.render()
        return {
            a : 'kek' for a in self.agents
        }, {
            a : 0 for a in self.agents
        }, {
            a : 'infou' for a in self.agents
        }, {
            a : False for a in self.agents 
        }, {
            a : False for a in self.agents
        }

    def observe(self, agent):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


