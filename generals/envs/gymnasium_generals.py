from copy import deepcopy
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from generals.agents import Agent, RandomAgent
from generals.core.game import Action, Game
from generals.core.grid import Grid, GridFactory
from generals.core.observation import Observation
from generals.core.replay import Replay
from generals.gui import GUI
from generals.gui.properties import GuiMode
from generals.rewards.reward_fn import RewardFn
from generals.rewards.win_lose_reward_fn import WinLoseRewardFn


class GymnasiumGenerals(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        grid_factory: GridFactory | None = None,
        npc: Agent | None = None,
        agent: Agent | None = None,  # Optional, just to obtain id and color
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ):
        self.render_mode = render_mode
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()
        self.reward_fn = reward_fn if reward_fn is not None else WinLoseRewardFn()
        # Observation for the agent at the prior time-step.
        self.prior_observation: None | Observation = None

        # Agents
        if npc is None:
            print('No NPC agent provided. Creating "Random" NPC as a fallback.')
            npc = RandomAgent()
        else:
            assert isinstance(npc, Agent), "NPC must be an instance of Agent class."
        self.npc = npc
        self.agent_id = "Agent" if agent is None else agent.id
        self.agent_ids = [self.agent_id, self.npc.id]
        self.agent_data = {
            self.agent_id: {"color": (67, 70, 86) if agent is None else agent.color},
            self.npc.id: {"color": self.npc.color},
        }
        assert self.agent_id != npc.id, "Agent ids must be unique - you can pass custom ids to agent constructors."

        # Game
        grid = self.grid_factory.generate()
        self.game = Game(grid, [self.agent_id, self.npc.id])
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
        max_army_value = 100_000
        max_timestep = 100_000
        max_land_value = np.prod(dims)
        grid_multi_binary = spaces.MultiBinary(dims)
        grid_discrete = np.ones(dims, dtype=int) * 100_000
        return spaces.Dict(
            {
                "armies": spaces.MultiDiscrete(grid_discrete),
                "generals": grid_multi_binary,
                "cities": grid_multi_binary,
                "mountains": grid_multi_binary,
                "neutral_cells": grid_multi_binary,
                "owned_cells": grid_multi_binary,
                "opponent_cells": grid_multi_binary,
                "fog_cells": grid_multi_binary,
                "structures_in_fog": grid_multi_binary,
                "owned_land_count": spaces.Discrete(max_land_value),
                "owned_army_count": spaces.Discrete(max_army_value),
                "opponent_land_count": spaces.Discrete(max_land_value),
                "opponent_army_count": spaces.Discrete(max_army_value),
                "timestep": spaces.Discrete(max_timestep),
                "priority": spaces.Discrete(2),
            }
        )

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
        self.game = Game(grid, self.agent_ids)

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

        observation = self.game.agent_observation(self.agent_id)
        info: dict[str, Any] = {}
        return observation, info

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        # Get action of NPC
        npc_observation = self.game.agent_observation(self.npc.id)
        npc_action = self.npc.act(npc_observation)
        actions = {self.agent_id: action, self.npc.id: npc_action}

        observations, infos = self.game.step(actions)

        # From observations of all agents, pick only those relevant for the main agent
        obs = observations[self.agent_id]
        info = infos[self.agent_id]
        if self.prior_observation is None:
            # Cannot compute a reward without a prior-observation. This should only happen
            # on the first time-step.
            reward = 0.0
        else:
            reward = self.reward_fn(
                prior_obs=self.prior_observation,
                # Technically, action is the prior-action, since it's what gives rise to the
                # current observation.
                prior_action=action,
                obs=obs,
            )

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

        self.prior_observation = observations[self.agent_id]
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.render_mode == "human":
            self.gui.close()
