import numpy as np
import gymnasium
import functools
from copy import deepcopy

from ..game import Game
from ..replay import Replay
from ..rendering import Renderer
from collections import OrderedDict


class Gym_Generals(gymnasium.Env):
    def __init__(self, mapper=None, agents=None, reward_fn=None, render_mode=None):
        self.render_mode = render_mode
        self.reward_fn = self.default_rewards if reward_fn is None else reward_fn
        self.mapper = mapper

        self.agent_name = agents[0].name
        self.npc = agents[1]

        self.agent_data = {agent.name: {"color": agent.color} for agent in agents}

        # check whether agents have unique names
        assert (
            agents[0].name != agents[1].name
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

    def render(self, tick_rate=None):
        if self.render_mode == "human":
            self.renderer.render()
            if tick_rate is not None:
                self.renderer.clock.tick(tick_rate)

    def reset(self, map: np.ndarray = None, seed=None, options={}):
        super().reset(seed=seed)
        # If map is not provided, generate a new one
        self.mapper.reset()
        map = self.mapper.get_map(numpify=True)

        self.game = Game(map, [self.agent_name, self.npc.name])
        self.npc.reset()

        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space

        if self.render_mode == "human":
            from_replay = "from_replay" in options and options["from_replay"] or False
            self.renderer = Renderer(self.game, self.agent_data, from_replay)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                map=self.mapper.get_map(numpify=False),
                agent_data=self.agent_colors,
            )
            self.replay.add_state(deepcopy(self.game.channels))

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
