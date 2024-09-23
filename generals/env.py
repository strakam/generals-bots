import numpy as np
import functools
import pettingzoo
import gymnasium
from copy import copy
from . import game, utils, config, agents
from .rendering import Renderer
from collections import OrderedDict


def pz_generals(game_config: config.GameConfig=config.GameConfig(), reward_fn=None, render_mode="none"):
    """
    Here we apply wrappers to the environment.
    """
    env = PZ_Generals(game_config, reward_fn=reward_fn, render_mode=render_mode)
    return env

def gym_generals(game_config: config.GameConfig=config.GameConfig(), reward_fn=None, render_mode="none"):
    """
    Here we apply wrappers to the environment.
    """
    env = Gym_Generals(game_config, reward_fn=reward_fn, render_mode=render_mode)
    return env


class PZ_Generals(pettingzoo.ParallelEnv):
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

    def render(self, tick_rate=None):
        if self.render_mode == "human":
            self.renderer.render()
            if tick_rate is not None:
                self.renderer.clock.tick(tick_rate)


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

        observations = OrderedDict({
            agent: self.game._agent_observation(agent) for agent in self.agents
        })

        infos = self.game.get_infos()
        return observations, infos

    def step(self, actions):
        if self.replay:
            self.action_history.append(actions)

        observations, infos = self.game.step(actions)

        truncated = {agent: False for agent in self.agents} # no truncation
        terminated = {agent: True if self.game.is_done() else False for agent in self.agents}
        rewards = self.reward_fn(observations)

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents = []
            # if replay is on, store the game
            if self.replay:
                utils.store_replay(self.game.map, self.action_history, self.replay)

        return OrderedDict(observations), rewards, terminated, truncated, infos
    
    def default_rewards(self, observations):
        """
        Calculate rewards for each agent.
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        rewards = {agent: 0 for agent in self.agents}
        game_ended = any(observations[agent]["is_winner"] for agent in self.agents)
        if game_ended:
            for agent in self.agents:
                if observations[agent]["is_winner"]:
                    rewards[agent] = 1
                else:
                    rewards[agent] = -1
        return rewards



class Gym_Generals(gymnasium.Env):
    def __init__(
        self,
        game_config=None,
        reward_fn=None,
        render_mode="none"
    ):
        self.render_mode = render_mode
        self.game_config = game_config
        self.reward_fn = self.default_rewards if reward_fn is None else reward_fn


        if game_config.agent_names is None:
            self.agent_name = "Player"
        else:
            self.agent_name = game_config.agent_names[0]

        self.opponent = agents.RandomAgent("Opponent")
        
        _map = utils.map_from_generator(
            grid_size=self.game_config.grid_size,
            mountain_density=self.game_config.mountain_density,
            city_density=self.game_config.city_density,
            general_positions=self.game_config.general_positions,
        )
        _game = game.Game(_map, [self.agent_name, "Opponent"])
        self.observation_space = _game.observation_space
        self.action_space = _game.action_space


    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return self.game.action_space

    def render(self, tick_rate=None):
        if self.render_mode == "human":
            self.renderer.render()
            if tick_rate is not None:
                self.renderer.clock.tick(tick_rate)


    def reset(self, map: np.ndarray = None, seed=None, options={}):
        # If map is not provided, generate a new one
        if map is None:
            map = utils.map_from_generator(
                grid_size=self.game_config.grid_size,
                mountain_density=self.game_config.mountain_density,
                city_density=self.game_config.city_density,
                general_positions=self.game_config.general_positions,
                seed=seed,
            )

        self.game = game.Game(map, [self.agent_name, "Opponent"])

        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space

        if self.render_mode == "human":
            self.renderer = Renderer(self.game)

        if "replay_file" in options:
            self.replay = options["replay_file"]
            self.action_history = []
        else:
            self.replay = False

        observation = OrderedDict(self.game._agent_observation(self.agent_name))
        info = self.game.get_infos()[self.agent_name]
        return observation, info

    def step(self, action):
        # get action of NPC
        npc_action = self.opponent.play(self.game._agent_observation("Opponent"))
        actions = {
            self.agent_name: action,
            "Opponent": npc_action
        }

        if self.replay:
            self.action_history.append(actions)

        observations, infos = self.game.step(actions)

        observation = observations[self.agent_name]
        info = infos[self.agent_name]
        truncated = False
        terminated = True if self.game.is_done() else False
        reward = self.reward_fn(observations)

        if terminated:
            # if replay is on, store the game
            if self.replay:
                utils.store_replay(self.game.map, self.action_history, self.replay)

        return OrderedDict(observation), reward, terminated, truncated, info
    
    def default_rewards(self, observations):
        """
        Calculate rewards for each agent.
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        reward = 0
        game_ended = any(observations[agent]["is_winner"] for agent in [self.agent_name, "Opponent"])
        if game_ended:
            reward = 1 if observations[self.agent_name]["is_winner"] else -1
        return reward
