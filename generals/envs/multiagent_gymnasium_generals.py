# type: ignore
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generals.core.action import Action, compute_valid_move_mask
from generals.core.game import Game
from generals.core.grid import Grid, GridFactory
from generals.core.observation import Observation
from generals.core.replay import Replay
from generals.gui import GUI
from generals.gui.properties import GuiMode
from generals.rewards.reward_fn import RewardFn
from generals.rewards.win_lose_reward_fn import WinLoseRewardFn


class MultiAgentGymnasiumGenerals(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        agents: list[str],
        grid_factory: GridFactory | None = None,
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()
        self.reward_fn = reward_fn if reward_fn is not None else WinLoseRewardFn()
        # Observation for the agent at the prior time-step.
        self.prior_observations: None | dict[str, Observation] = None

        self.agents = agents
        self.colors = [(255, 107, 108), (0, 130, 255)]
        self.agent_data = {id: {"color": color} for id, color in zip(agents, self.colors)}

        # Game
        grid = self.grid_factory.generate()
        self.game = Game(grid, agents)
        self.truncation = truncation
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

    def set_observation_space(self) -> spaces.Space:
        """
        If grid_factory has padding on, grid (and therefore observations) will be padded to the same shape,
        which corresponds to the maximum grid dimensions of grid_factory.
        Otherwise, the observatoin shape might change depending on the currently generated grid.

        Note: The grid is padded with mountains from right and bottom. We recommend using the padded
        grids for training purposes, as it will make the observations consistent across episodes.
        """
        if self.grid_factory.padding:
            dims = self.grid_factory.max_grid_dims
        else:
            dims = self.game.grid_dims

        return spaces.Box(low=0, high=2**31 - 1, shape=(2, 15, dims[0], dims[1]), dtype=np.float32)

    def set_action_space(self) -> spaces.Space:
        if self.grid_factory.padding:
            dims = self.grid_factory.max_grid_dims
        else:
            dims = self.game.grid_dims
        return spaces.MultiDiscrete([2, dims[0], dims[1], 4, 2])

    def render(self):
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        if options is None:
            options = {}

        if "grid" in options:
            grid = Grid(options["grid"])
        else:
            # Provide the np.random.Generator instance created in Env.reset()
            # as opposed to creating a new one with the same seed.
            self.grid_factory.set_rng(rng=self.np_random)
            grid = self.grid_factory.generate()

        # Create game for current run
        self.game = Game(grid, self.agents)

        # Create GUI for current render run
        if self.render_mode == "human":
            self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)

        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        _obs = {agent: self.game.agent_observation(agent) for agent in self.agents}
        observations = np.stack([_obs[agent].as_tensor() for agent in self.agents], dtype=np.float32)

        infos: dict[str, Any] = self.game.get_infos()
        # flatten infos
        infos = {
            agent: [
                infos[agent]["army"],
                infos[agent]["land"],
                infos[agent]["is_done"],
                infos[agent]["is_winner"],
                compute_valid_move_mask(_obs[agent]),
            ]
            for i, agent in enumerate(self.agents)
        }
        return observations, infos

    def step(self, actions: list[Action]) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        _actions = {
            self.agents[0]: actions[0],
            self.agents[1]: actions[1],
        }

        observations, infos = self.game.step(_actions)
        obs1 = self.game.agent_observation(self.agents[0]).as_tensor()
        obs2 = self.game.agent_observation(self.agents[1]).as_tensor()
        obs = np.stack([obs1, obs2])

        # flatten infos
        infos = {
            agent: [
                infos[agent]["army"],
                infos[agent]["land"],
                infos[agent]["is_done"],
                infos[agent]["is_winner"],
                compute_valid_move_mask(observations[agent]),
            ]
            for agent in self.agents
        }

        # From observations of all agents, pick only those relevant for the main agent
        if self.prior_observations is None:
            # Cannot compute a reward without a prior-observation. This should only happen
            # on the first time-step.
            rewards = [0.0, 0.0]
        else:
            rewards = [
                self.reward_fn(
                    prior_obs=self.prior_observations[agent],
                    # Technically, action is the prior-action, since it's what gives rise to the
                    # current observation.
                    prior_action=_actions[agent],
                    obs=observations[agent],
                )
                for agent in self.agents
            ]
        rewards = 0 # WIP

        terminated = self.game.is_done()
        truncated = False
        if self.truncation is not None:
            truncated = self.game.time >= self.truncation

        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))

        if terminated or truncated:
            if hasattr(self, "replay"):
                self.replay.store()

        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()

        self.prior_observations = {agent: observations[agent] for agent in self.agents}
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
