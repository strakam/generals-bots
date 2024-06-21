import numpy as np
import functools
from copy import copy


import pettingzoo

from typing import List

from . import game, rendering, utils


def generals_v0(map: np.ndarray, render_mode="human"):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(map, render_mode=render_mode)
    return env


class Generals(pettingzoo.ParallelEnv):
    metadata = {"render.modes": ["human", "none"]}

    def __init__(
        self,
        map: np.ndarray,
        agent_names: List[str] = ["red", "blue"],
        render_mode="human",
    ):
        self.render_mode = render_mode

        self.map = map
        self.agents = agent_names
        self.possible_agents = self.agents[:]

        if render_mode == "human":
            rendering.init_screen(
                agents=self.agents,
                grid_size=self.map.shape[0],
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name):
        assert agent_name in self.agents, f"{agent_name} is not a valid agent"
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name):
        assert agent_name in self.agents, f"{agent_name} is not a valid agent"
        return self.game.action_space

    def render(self):
        if self.render_mode == "human":
            rendering.render(self.game)

    def reset(self, seed=None, options={}):
        self.agents = copy(self.possible_agents)
        self.game = game.Game(self.map, self.possible_agents)

        if "replay" in options:
            self.replay = options["replay"]
            self.action_history = []
        else:
            self.replay = False

        observations = {
            agent: self.game._agent_observation(agent) for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.game.step(actions)

        if self.replay:
            self.action_history.append(actions)

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            terminated = {agent: True for agent in self.agents}
            self.agents = []
            # if replay is on, store the game
            if self.replay:
                utils.store_replay(self.game.map, self.action_history, self.replay)

        return observations, rewards, terminated, truncated, infos

    def close(self):
        rendering.pygame.quit()
