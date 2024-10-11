import functools
from collections.abc import Callable
from typing import TypeAlias

import pettingzoo
from gymnasium import spaces
from copy import deepcopy

from generals.core.game import Action, Observation, Info
from generals.core.grid import GridFactory

from generals.envs.common_environment import CommonEnv

# Type aliases
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[dict[str, Observation], Action, bool, Info], Reward]
AgentID: TypeAlias = str


class PettingZooGenerals(pettingzoo.ParallelEnv, CommonEnv):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }
    default_colors = [
        (67, 70, 86),
        (242, 61, 106),
        (0, 255, 0),
        (0, 0, 255),
    ]  # Up for improvement (needs to be extended for multiple agents)

    def __init__(
        self,
        grid_factory: GridFactory,
        agent_ids: list[str],
        render_mode=None,
    ):
        CommonEnv.__init__(self, grid_factory, render_mode)

        self.agent_data = {
            agent_id: {"color": color}
            for agent_id, color in zip(agent_ids, self.default_colors)
        }
        self.agent_ids = agent_ids
        self.possible_agents = agent_ids

        assert len(self.possible_agents) == len(
            set(self.possible_agents)
        ), "Agent ids must be unique - you can pass custom ids to agent constructors."

        self.reward_fn = self._default_reward

    @property
    def reward_fn(self) -> RewardFn:
        return self._reward_fn

    @reward_fn.setter
    def reward_fn(self, rew_fn: RewardFn):
        self._reward_fn = rew_fn

    def set_color(self, agent_id: str, color: tuple[int, int, int]) -> None:
        self.agent_data[agent_id]["color"] = color

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> spaces.Space:
        assert agent in self.possible_agents, f"Agent {agent} not in possible agents"
        return self.game.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> spaces.Space:
        assert agent in self.possible_agents, f"Agent {agent} not in possible agents"
        return self.game.action_space

    def render(self, fps: int = None) -> None:
        fps = self.metadata["render_fps"] if fps is None else fps
        if self.render_mode == "human":
            _ = self.gui.tick(fps=fps)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, Observation], dict[AgentID, dict]]:
        if options is None:
            options = {}
        self.agents = deepcopy(self.possible_agents)
        observations, infos = self._reset(seed, options)
        return observations, infos

    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        dict[AgentID, Observation],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, Info],
    ]:
        observations, rewards, terminated, truncated, infos = self._step(actions)
        return observations, rewards, terminated, truncated, infos

    @staticmethod
    def _default_reward(
        observation: dict[str, Observation],
        action: Action,
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

    def close(self) -> None:
        self.gui.close()
