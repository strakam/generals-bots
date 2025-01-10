# type: ignore
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generals.core.action import Action, compute_valid_move_mask
from generals.core.environment import Environment
from generals.core.grid import GridFactory
from generals.core.observation import Observation
from generals.rewards.reward_fn import RewardFn


class GymnasiumGenerals(gym.Env):
    """ """

    def __init__(
        self,
        agent_ids: list[str],
        grid_factory: GridFactory = None,
        truncation: int = None,
        reward_fn: RewardFn = None,
        to_render: bool = False,
        speed_multiplier: float = 1.0,
        save_replays: bool = False,
        pad_to: int = 24,
    ):
        self.agent_ids = agent_ids
        self.environment = Environment(
            agent_ids, grid_factory, truncation, reward_fn, to_render, speed_multiplier, save_replays, pad_to
        )
        self.pad_to = pad_to
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

    def set_observation_space(self) -> spaces.Space:
        return spaces.Box(low=0, high=2**31 - 1, shape=(2, 15, self.pad_to, self.pad_to), dtype=np.float32)

    def set_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete([2, self.pad_to, self.pad_to, 4, 2])

    def render(self):
        self.environment.render()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        # Reset the parent Gymnasium.env first.
        super().reset(seed=seed)
        # Provide the np.random.Generator instance created in Env.reset()
        # as opposed to creating a new one with the same seed.
        _obs, _infos = self.environment.reset_from_gymnasium(rng=self.np_random, options=options)
        _rewards = {agent: 0 for agent in self.agent_ids}
        obs = self.flatten_obs(_obs)
        infos = self.flatten_infos(_obs, _infos, _rewards)
        return obs, infos

    def step(self, flat_actions: np.ndarray) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        actions = self.deflate_actions(flat_actions)
        _obs, _rewards, terminated, truncated, _infos = self.environment.step(actions)

        obs = self.flatten_obs(_obs)
        infos = self.flatten_infos(_obs, _infos, _rewards)
        return obs, 0, terminated, truncated, infos

    def deflate_actions(self, action: np.ndarray) -> dict[str, Action]:
        return {agent: Action(*action[i]) for i, agent in enumerate(self.agent_ids)}

    def flatten_obs(self, obs: dict[str, Observation]) -> np.ndarray:
        return np.stack(
            [
                obs[self.agent_ids[0]].as_tensor(),
                obs[self.agent_ids[1]].as_tensor(),
            ]
        )

    def flatten_infos(
        self, obs: dict[str, Observation], infos: dict[str, Any], rewards: dict[str, float]
    ) -> dict[str, Any]:
        return {
            agent: [
                rewards[agent],
                infos[agent]["army"],
                infos[agent]["land"],
                infos[agent]["is_winner"],
                compute_valid_move_mask(obs[agent]),
            ]
            for i, agent in enumerate(self.agent_ids)
        }

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
