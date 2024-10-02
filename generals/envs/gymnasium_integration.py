from collections.abc import Callable
from typing import TypeAlias, Any, SupportsFloat

import gymnasium as gym
import functools
from copy import deepcopy

from generals.agents import Agent
from generals.core.game import Game, Action, Observation, Info
from generals.core.grid import GridFactory
from generals.gui import GUI
from generals.core.replay import Replay

# Type aliases
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[dict[str, Observation], Action, bool, Info], Reward]


class Gym_Generals(gym.Env):
    def __init__(
        self,
        grid_factory: GridFactory,
        agent: Agent,
        npc: Agent,
        reward_fn: RewardFn = None,
        render_mode=None,
    ):
        self.game = None
        self.gui = None
        self.replay = None

        self.render_mode = render_mode
        self.reward_fn = self._default_reward if reward_fn is None else reward_fn
        self.grid_factory = grid_factory

        self.agent_name = agent.name
        self.npc = npc

        self.agent_data = {agent.name: {"color": agent.color} for agent in [agent, npc]}

        # check whether agents have unique names
        assert (
            agent.name != npc.name
        ), "Agent names must be unique - you can pass custom names to agent constructors."

        grid = self.grid_factory.grid_from_generator()
        game = Game(grid, [self.agent_name, self.npc.name])
        self.observation_space = game.observation_space
        self.action_space = game.action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self) -> gym.Space:
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self) -> gym.Space:
        return self.game.action_space

    def render(self, fps: int = 6) -> None:
        if self.render_mode == "human":
            self.gui.tick(fps=fps)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        Observation, dict[str, Any]
    ]:
        if options is None:
            options = {}
        super().reset(seed=seed)
        # If map is not provided, generate a new one
        if "grid" in options:
            grid = self.grid_factory.grid_from_string(options["grid"])
        else:
            grid = self.grid_factory.grid_from_generator()

        self.game = Game(grid, [self.agent_name, self.npc.name])
        self.npc.reset()

        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space

        if self.render_mode == "human":
            self.gui = GUI(self.game, self.agent_data)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        observation = self.game._agent_observation(self.agent_name)
        info = {}
        return observation, info

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
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
        done = terminated or truncated
        reward = self.reward_fn(observation, action, done, info)

        if terminated:
            if hasattr(self, "replay"):
                self.replay.store()

        return observation, reward, terminated, truncated, info

    @staticmethod
    def _default_reward(
        observation: dict[str, Observation],
        action: Action,
        done: bool,
        info: Info,
    ) -> Reward:
        """
        Calculate rewards for each agent.
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        if done:
            reward = 1 if observation["observation"]["is_winner"] else -1
        else:
            reward = 0
        return reward
