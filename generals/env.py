import functools
from copy import copy
import time

import gymnasium
from gymnasium.spaces import Discrete

import pettingzoo
from pettingzoo.utils import wrappers, agent_selector

from typing import List

from . import game, config, utils




def generals_v0(config=config.Config, render_mode="human"):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(config)
    # Apply parallel_to_aec to support AEC api
    # env = pettingzoo.utils.parallel_to_aec(env)
    return env


class Generals(pettingzoo.ParallelEnv):
    metadata = {'render.modes': ["human", "none"]}

    def __init__(self, game_config: config.Config, render_mode="human"):
        self.game_config = game_config
        self.render_mode = render_mode

        agent_names = ['red', 'blue']
        self.agent_to_id = {agent: i+1 for i, agent in enumerate(agent_names)}
        self.id_to_agent = {i+1: agent for i, agent in enumerate(agent_names)}
        self.possible_agents = [i+1 for i in range(len(agent_names))]

        if render_mode == "human":
            utils.init_screen(self.game_config)

    def observation_space(self, agent):
        observation = self.game.agent_observation(agent)
        return observation

    # Action space should be defined here.
    def action_space(self, agent):
        valid_actions = self.game.valid_actions(agent, view='list')
        return valid_actions


    def render(self):
        if self.render_mode == "human":
            utils.handle_events()
            utils.render_grid(self.game, [1, 2])
            utils.render_gui(self.game, self.id_to_agent)
            utils.pygame.display.flip()
            time.sleep(0.5) # this is digsuting, fix it later (?)

    def reset(self, seed=None, options=None):
        self.game = game.Game(self.game_config)
        self.agents = copy(self.possible_agents)
        self.state = {agent: None for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.observations = {a : 'kek' for a in self.agents}
        self.rewards = {a : 0 for a in self.agents}
        self.termination = {a : False for a in self.agents}
        self.truncation = {a : False for a in self.agents}
        self.infos = {agent : self.game for agent in self.agents}

        observations = {
            a : 'kek' for a in self.agents
        }

        infos = {
            a : 'infou' for a in self.agents
        }

        return self.observations, self.rewards, self.termination, self.truncation, self.infos


    def step(self, actions):
        # return some dummy values for now
        self.game.step(actions)
        self.state = {agent: None for agent in self.agents}
        self.render()
        self.observations = {a : 'kek' for a in self.agents}
        self.rewards = {a : 0 for a in self.agents}
        self.termination = {a : False for a in self.agents}
        self.truncation = {a : False for a in self.agents}
        self.infos = {agent : self.game for agent in self.agents}
        self.agent_selection = self._agent_selector.next()

        return self.observations, self.rewards, self.termination, self.truncation, self.infos

    def observe(self, agent):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass


