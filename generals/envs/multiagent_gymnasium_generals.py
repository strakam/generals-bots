# type: ignore
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generals.core.action import Action
from generals.core.environment import Environment
from generals.core.grid import GridFactory
from generals.core.observation import Observation
from generals.rewards.reward_fn import RewardFn


class MultiAgentGymnasiumGenerals(gym.Env):
    def __init__(
        self,
        agent_ids: list[str],
        grid_factory: GridFactory = None,
        truncation: int = None,
        reward_fn: RewardFn = None,
        to_render: bool = False,
        save_replays: bool = False,
    ):
        self.environment = Environment(agent_ids, grid_factory, truncation, reward_fn, to_render, save_replays)
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

    def set_observation_space(self) -> spaces.Space:
        dims = self.environment.grid_dims
        return spaces.Box(low=0, high=2**31 - 1, shape=(2, 15, dims[0], dims[1]), dtype=np.float32)

    def set_action_space(self) -> spaces.Space:
        dims = self.environment.grid_dims
        return spaces.MultiDiscrete([2, dims[0], dims[1], 4, 2])

    def render(self):
        self.environment.render()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        # Reset the parent Gymnasium.env first.
        super().reset(seed=seed)
        # Provide the np.random.Generator instance created in Env.reset()
        # as opposed to creating a new one with the same seed.
        observation, info = self.environment.reset_from_gymnasium(rng=self.np_random, options=options)

        return observation, info

    def step(self, agent_id_to_action: dict[str, Action]) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self.environment.step(agent_id_to_action)
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
