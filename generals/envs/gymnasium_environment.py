from collections.abc import Callable
from typing import TypeAlias, Any, SupportsFloat

import gymnasium as gym
import functools

from generals.agents import Agent
from generals.core.game import Game, Action, Observation, Info
from generals.core.grid import GridFactory

from generals.envs.common_environment import CommonEnv

# Type aliases
Reward: TypeAlias = float
RewardFn: TypeAlias = Callable[[dict[str, Observation], Action, bool, Info], Reward]


class GymnasiumGenerals(CommonEnv):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        grid_factory: GridFactory = None,
        npc: Agent = None,
        render_mode=None,
        agent_id: str = "Agent",
        agent_color: tuple[int, int, int] = (67, 70, 86),
    ):
        CommonEnv.__init__(self, grid_factory, render_mode)

        # Agents
        assert isinstance(npc, Agent), "NPC must be an instance of Agent class."
        self.npc = npc
        self.agent_id = agent_id
        self.agent_ids = [agent_id, self.npc.id]
        self.agent_data = {
            agent_id: {"color": agent_color},
            self.npc.id: {"color": self.npc.color},
        }
        assert (
            agent_id != npc.id
        ), "Agent ids must be unique - you can pass custom ids to agent constructors."

        # Game
        grid = self.grid_factory.grid_from_generator()
        game = Game(grid, [self.agent_id, self.npc.id])
        self.observation_space = game.observation_space
        self.action_space = game.action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self) -> gym.Space:
        return self.observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self) -> gym.Space:
        return self.action_space

    def render(self):
        self._render()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        if options is None:
            options = {}
        _obs, _info = self._reset(seed, options)

        self.observation_space = self.game.observation_space
        self.action_space = self.game.action_space

        observation = _obs[self.agent_id]
        info = _info[self.agent_id]
        return observation, info

    def step(
        self, action: Action
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Get action of NPC
        npc_action = self.npc.act(self.game._agent_observation(self.npc.id))
        actions = {self.agent_id: action, self.npc.id: npc_action}

        _obs, _rew, _term, _trunc, _info = self._step(actions)
        observation = _obs[self.agent_id]
        reward = _rew[self.agent_id]
        terminated = _term[self.agent_id]
        truncated = _trunc[self.agent_id]
        info = _info[self.agent_id]
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

    def close(self) -> None:
        if hasattr(self, "gui"):
            self.gui.close()
