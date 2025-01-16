from copy import deepcopy
from dataclasses import dataclass
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


@dataclass
class AgentInfo:
    """Data structure to hold agent-specific information."""

    color: tuple[int, int, int]


class GymnasiumGenerals(gym.Env):
    """A Gymnasium environment for the Generals game.

    This environment implements a two-player strategic game where agents compete
    for territory and resources on a grid-based map.
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 6,
    }

    def __init__(
        self,
        agents: list[str],
        grid_factory: GridFactory | None = None,
        pad_observations_to: int = 24,
        truncation: int | None = None,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ):
        """Initialize the Generals environment.

        Args:
            agents: List of agent identifiers
            grid_factory: Factory for generating game grids
            truncation: Maximum number of steps before truncation
            reward_fn: Function for computing rewards
            render_mode: Visualization mode ('human' or None)
        """
        # Initialize basic parameters
        self.render_mode = render_mode
        self.grid_factory = grid_factory or GridFactory()
        self.reward_fn = reward_fn or WinLoseRewardFn()
        self.agents = agents
        self.truncation = truncation
        self.pad_observations_to = pad_observations_to

        # Initialize agent-specific data
        self.agent_data = self._setup_agent_data()

        # Initialize game state
        self.prior_observations: dict[str, Observation] | None = None
        grid = self.grid_factory.generate()
        self.game = Game(grid, self.agents)

        # Set up spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()

    def _setup_agent_data(self) -> dict[str, dict[str, Any]]:
        """Set up initial data for each agent."""
        colors = [(255, 107, 108), (0, 130, 255)]
        return {id: {"color": color} for id, color in zip(self.agents, colors)}

    def _create_observation_space(self) -> spaces.Space:
        """Create the observation space based on grid dimensions."""
        dim = self.pad_observations_to
        return spaces.Box(low=0, high=2**31 - 1, shape=(2, 15, dim, dim), dtype=np.float32)

    def _create_action_space(self) -> spaces.Space:
        """Create the action space based on grid dimensions."""
        dim = self.pad_observations_to
        return spaces.MultiDiscrete([2, dim, dim, 4, 2])

    def _process_observations(self, observations: dict[str, Observation]) -> np.ndarray:
        """Process raw observations into the required tensor format."""
        processed_obs = []
        for agent in self.agents:
            observations[agent].pad_observation(pad_to=self.pad_observations_to)
            processed_obs.append(observations[agent].as_tensor())
        return np.stack(processed_obs)

    def _process_infos(self, observations: dict[str, Observation], game_infos: dict[str, Any]) -> dict[str, list]:
        """Process game information into a structured format."""
        return {
            agent: [
                game_infos[agent]["army"],
                game_infos[agent]["land"],
                game_infos[agent]["is_done"],
                game_infos[agent]["is_winner"],
                compute_valid_move_mask(observations[agent]),
            ]
            for agent in self.agents
        }

    def _compute_rewards(self, actions: dict[str, Action], observations: dict[str, Observation]) -> list[float]:
        """Compute rewards for all agents based on their actions and observations."""
        if self.prior_observations is None:
            return [0.0, 0.0]

        return [
            self.reward_fn(
                prior_obs=self.prior_observations[agent],
                prior_action=actions[agent],
                obs=observations[agent],
            )
            for agent in self.agents
        ]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        options = options or {}

        # Initialize grid
        if "grid" in options:
            grid = Grid(options["grid"])
        else:
            self.grid_factory.set_rng(rng=self.np_random)
            grid = self.grid_factory.generate()

        # Create new game instance
        self.game = Game(grid, self.agents)

        # Setup visualization if needed
        if self.render_mode == "human":
            self.gui = GUI(self.game, self.agent_data, GuiMode.TRAIN)

        # Handle replay functionality
        if "replay_file" in options:
            self.replay = Replay(
                name=options["replay_file"],
                grid=grid,
                agent_data=self.agent_data,
            )
            self.replay.add_state(deepcopy(self.game.channels))
        elif hasattr(self, "replay"):
            del self.replay

        # Get and process observations
        raw_obs = {agent: self.game.agent_observation(agent) for agent in self.agents}
        observations = self._process_observations(raw_obs)
        infos = self._process_infos(raw_obs, self.game.get_infos())

        return observations, infos

    def step(self, actions: list[Action]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one time step within the environment."""
        # Convert actions list to dictionary
        action_dict = {self.agents[i]: action for i, action in enumerate(actions)}

        # Execute game step
        observations, infos = self.game.step(action_dict)

        # Process observations and info
        processed_obs = self._process_observations(observations)
        processed_infos = self._process_infos(observations, infos)

        # Compute rewards (currently WIP)
        rewards = 0  # placeholder for WIP reward system

        # Check termination conditions
        terminated = self.game.is_done()
        truncated = False if self.truncation is None else self.game.time >= self.truncation

        # Handle replay if active
        if hasattr(self, "replay"):
            self.replay.add_state(deepcopy(self.game.channels))
            if terminated or truncated:
                self.replay.store()

        self.prior_observations = {agent: observations[agent] for agent in self.agents}

        return processed_obs, rewards, terminated, truncated, processed_infos

    def render(self) -> None:
        """Render the game state."""
        if self.render_mode == "human":
            _ = self.gui.tick(fps=self.metadata["render_fps"])

    def close(self) -> None:
        """Clean up resources."""
        if self.render_mode == "human":
            self.gui.close()
