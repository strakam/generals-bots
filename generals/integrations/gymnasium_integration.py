import numpy as np
import gymnasium
import functools
from copy import deepcopy

from ..game import Game
from ..replay import Replay
from ..rendering import Renderer
from collections import OrderedDict


class Gym_Generals(gymnasium.Env):
    def __init__(self, mapper, agent, npc, reward_fn=None, render_mode=None):
        self.render_mode = render_mode
        self.reward_fn = self.default_rewards if reward_fn is None else reward_fn
        self.mapper = mapper

        self.agent_name = agent.name
        self.npc = npc

        self.agent_data = {agent.name: {"color": agent.color} for agent in [agent, npc]}

        # check whether agents have unique names
        assert (
            agent.name != npc.name
        ), "Agent names must be unique - you can pass custom names to agent constructors."

        map = self.mapper.get_map(numpify=True)
        game = Game(map, [self.agent_name, self.npc.name])
        self.observation_space = game.observation_space
        self.action_space = game.action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return self.game.action_space

    def render(self, fps=6):
        if self.render_mode == "human":
            self.renderer.render(fps=fps)

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        # If map is not provided, generate a new one
        if "map" in options:
            map = options["map"]
        else:
            self.mapper.reset() # Generate new map
            map = self.mapper.get_map()

        self.game = Game(self.mapper.numpify_map(map), [self.agent_name, self.npc.name])
        self.npc.reset()

        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space

        if self.render_mode == "human":
            self.renderer = Renderer(self.game, self.agent_data)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                map=map,
                agent_data=self.agent_colors,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        observation = OrderedDict(self.game._agent_observation(self.agent_name))
        info = {}
        return observation, info

    def step(self, action):
        # get action of NPC
        npc_action = self.npc.play(self.game._agent_observation(self.npc.name))
        actions = {self.agent_name: action, self.npc.name: npc_action}

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        observations, infos = self.game.step(actions)

        observation = observations[self.agent_name]
        info = infos[self.agent_name]
        truncated = False
        terminated = True if self.game.is_done() else False
        reward = self.reward_fn(observations)

        if terminated:
            if hasattr(self, "replay"):
                self.replay.save()

        return OrderedDict(observation), reward, terminated, truncated, info

    def default_rewards(self, observations):
        """
        Calculate rewards for each agent.
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        reward = 0
        game_ended = any(
            observations[agent]["is_winner"]
            for agent in [self.agent_name, self.npc.name]
        )
        if game_ended:
            reward = 1 if observations[self.agent_name]["is_winner"] else -1
        return reward
