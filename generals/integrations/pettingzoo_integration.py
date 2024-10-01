import functools
from collections.abc import Callable
from typing import TypeAlias

import pettingzoo
from gymnasium import spaces
from copy import deepcopy
from ..game import Game, Observation
from ..grid import GridFactory
from ..agents import Agent
from ..replay import Replay
from ..rendering import Renderer


# Type aliases
from generals.game import Info

Reward: TypeAlias = dict[str, float]
RewardFn: TypeAlias = Callable[[dict[str, Observation]], Reward]


class PZ_Generals(pettingzoo.ParallelEnv):
    def __init__(
        self,
        grid_factory: GridFactory,
        agents: dict[str, Agent],
        reward_fn: RewardFn = None,
        render_mode=None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory

        self.agent_data = {
            agents[agent].name: {"color": agents[agent].color}
            for agent in agents.keys()
        }
        self.possible_agents = list(agents.keys())

        assert (
            len(self.possible_agents) == len(set(self.possible_agents))
        ), "Agent names must be unique - you can pass custom names to agent constructors."

        self.reward_fn = self.default_reward if reward_fn is None else reward_fn

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
        self.agents = deepcopy(self.possible_agents)

        if "grid" in options:
            grid = self.grid_factory.grid_from_string(options["grid"])
        else:
            grid = self.grid_factory.grid_from_generator(seed=seed)

        self.game = Game(grid, self.agents)

        if self.render_mode == "human":
            self.renderer = Renderer(self.game, self.agent_data)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        observations = self.game.get_all_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action):
        observations, infos = self.game.step(action)

        truncated = {agent: False for agent in self.agents}  # no truncation
        terminated = {
            agent: True if self.game.is_done() else False for agent in self.agents
        }

        rewards = {
            agent: self.reward_fn(
                observations[agent],
                action,
                terminated[agent] or truncated[agent],
                infos[agent],
            )
            for agent in self.agents
        }

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        # if any agent dies, all agents are terminated
        terminate = any(terminated.values())
        if terminate:
            self.agents = []
            if hasattr(self, "replay"):
                self.replay.store()

        return observations, rewards, terminated, truncated, infos

    def default_reward(
        self,
        observation: dict[str, Observation],
        action: spaces.Tuple,
        done: bool,
        info: Info,
    ) -> Reward:
        """
        Give 0 if game still running, otherwise 1 for winner and -1 for loser.
        """
        if done:
            reward = 1 if observation["observation"]["is_winner"] else -1
        else:
            reward = 0
        return reward

    def close(self):
        print("Closing environment")
