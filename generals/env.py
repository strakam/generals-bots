import numpy as np
import functools
import pettingzoo
from copy import copy
from . import game, utils, config
from .rendering import Renderer


def generals_v0(game_config: config.GameConfig=config.GameConfig(), reward_fn=None, render_mode="none"):
    """
    Here we apply wrappers to the environment.
    """
    env = Generals(game_config, reward_fn=reward_fn, render_mode=render_mode)
    return env


class Generals(pettingzoo.ParallelEnv):
    def __init__(
        self,
        game_config = None,
        reward_fn=None,
        render_mode="none"
    ):
        self.render_mode = render_mode
        self.game_config = game_config
        self.agents = game_config.agent_names
        self.possible_agents = self.agents[:]

        self.reward_fn = self.default_rewards if reward_fn is None else reward_fn


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

        # If map is not provided, generate a new one
        if map is None:
            map = utils.map_from_generator(
                grid_size=self.game_config.grid_size,
                mountain_density=self.game_config.mountain_density,
                city_density=self.game_config.city_density,
                general_positions=self.game_config.general_positions,
                seed=seed,
            )

        self.game = game.Game(map, self.possible_agents)

        if self.render_mode == "human":
            self.renderer = Renderer(self.game)

        if "replay_file" in options:
            self.replay = options["replay_file"]
            self.action_history = []
        else:
            self.replay = False

        observations = {
            agent: self.game._agent_observation(agent) for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        if self.replay:
            self.action_history.append(actions)

        observations, infos = self.game.step(actions)

        truncated = {agent: False for agent in self.agents} # no truncation
        terminated = {agent: True if self.game.is_done() else False for agent in self.agents}
        rewards = self.reward_fn(observations, infos)

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents = []
            # if replay is on, store the game
            if self.replay:
                utils.store_replay(self.game.map, self.action_history, self.replay)

        return observations, rewards, terminated, truncated, infos
    
    def default_rewards(self, observations, infos):
        """
        Calculate rewards for each agent.
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        rewards = {agent: 0 for agent in self.agents}
        game_ended = any(infos[agent]["is_winner"] for agent in self.agents)
        if game_ended:
            for agent in self.agents:
                if infos[agent]["is_winner"]:
                    rewards[agent] = 1
                else:
                    rewards[agent] = -1
        return rewards
