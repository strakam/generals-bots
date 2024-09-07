import numpy as np
import functools
import pettingzoo
from typing import List
from copy import copy
from . import game, utils
from .rendering import Renderer
from pydantic import BaseModel

class GameConfig(BaseModel):
    grid_size: int = 16
    mountain_density: float = 0.2
    town_density: float = 0.05
    general_positions: List[str] = None
    map: str = None
    replay_file: str = None

def generals_v0(game_config: GameConfig, render_mode="human"):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(game_config, render_mode=render_mode)
    return env


class Generals(pettingzoo.ParallelEnv):
    def __init__(
        self,
        game_config = None,
        render_mode="human",
    ):
        self.render_mode = render_mode
        self.game_config = game_config
        self.agents = ["red", "blue"]
        self.possible_agents = self.agents[:]


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
            self.renderer.render()

    def reset(self, map: np.ndarray = None, seed=None, options={}):
        self.agents = copy(self.possible_agents)

        if map is None:
            map = utils.map_from_generator(
                grid_size=self.game_config.grid_size,
                mountain_density=self.game_config.mountain_density,
                town_density=self.game_config.town_density,
                general_positions=self.game_config.general_positions,
            )

        self.game = game.Game(map, self.possible_agents)

        if self.render_mode == "human":
            self.renderer = Renderer(self.game)

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
