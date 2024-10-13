import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.ndimage import maximum_filter  # type: ignore

from .channels import Channels
from .config import Action, Direction, Info, Observation
from .grid import Grid

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class Game:
    def __init__(self, grid: Grid, agents: list[str]):
        # Agents
        self.agents = agents

        # Grid
        _grid = grid.grid
        self.channels = Channels(_grid, self.agents)
        self.grid_dims = (_grid.shape[0], _grid.shape[1])
        self.general_positions = {
            agent: np.argwhere(_grid == chr(ord("A") + i))[0] for i, agent in enumerate(self.agents)
        }

        # Time stuff
        self.time = 0
        self.increment_rate = 50

        # Limits
        self.max_army_value = 10_000
        self.max_land_value = np.prod(self.grid_dims)
        self.max_timestep = 100_000

        # Spaces
        grid_multi_binary = gym.spaces.MultiBinary(self.grid_dims)
        grid_discrete = np.ones(self.grid_dims, dtype=int) * self.max_army_value
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "army": gym.spaces.MultiDiscrete(grid_discrete),
                        "general": grid_multi_binary,
                        "city": grid_multi_binary,
                        "owned_cells": grid_multi_binary,
                        "opponent_cells": grid_multi_binary,
                        "neutral_cells": grid_multi_binary,
                        "visible_cells": grid_multi_binary,
                        "structure": grid_multi_binary,
                        "owned_land_count": gym.spaces.Discrete(self.max_army_value),
                        "owned_army_count": gym.spaces.Discrete(self.max_army_value),
                        "opponent_land_count": gym.spaces.Discrete(self.max_army_value),
                        "opponent_army_count": gym.spaces.Discrete(self.max_army_value),
                        "is_winner": gym.spaces.Discrete(2),
                        "timestep": gym.spaces.Discrete(self.max_timestep),
                    }
                ),
                "action_mask": gym.spaces.MultiBinary(self.grid_dims + (4,)),
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "pass": gym.spaces.Discrete(2),
                "cell": gym.spaces.MultiDiscrete(list(self.grid_dims)),
                "direction": gym.spaces.Discrete(4),
                "split": gym.spaces.Discrete(2),
            }
        )

    def action_mask(self, agent: str) -> np.ndarray:
        """
        Function to compute valid actions from a given ownership mask.

        Valid action is an action that originates from agent's cell with atleast 2 units
        and does not bump into a mountain or fall out of the grid.
        Returns:
            np.ndarray: an NxNx4 array, where each channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.

            I.e. valid_action_mask[i, j, k] is 1 if action k is valid in cell (i, j).
        """

        ownership_channel = self.channels.ownership[agent]
        more_than_1_army = (self.channels.army > 1) * ownership_channel
        owned_cells_indices = self.channel_to_indices(more_than_1_army)
        valid_action_mask = np.zeros((self.grid_dims[0], self.grid_dims[1], 4), dtype=bool)

        if self.is_done():
            return valid_action_mask

        for channel_index, direction in enumerate(DIRECTIONS):
            destinations = owned_cells_indices + direction.value

            # check if destination is in grid bounds
            in_first_boundary = np.all(destinations >= 0, axis=1)
            in_height_boundary = destinations[:, 0] < self.grid_dims[0]
            in_width_boundary = destinations[:, 1] < self.grid_dims[1]
            destinations = destinations[in_first_boundary & in_height_boundary & in_width_boundary]

            # check if destination is road
            passable_cell_indices = self.channels.passable[destinations[:, 0], destinations[:, 1]] == 1
            action_destinations = destinations[passable_cell_indices]

            # get valid action mask for a given direction
            valid_source_indices = action_destinations - direction.value
            valid_action_mask[valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index] = 1.0
        return valid_action_mask

    def channel_to_indices(self, channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.
        """
        return np.argwhere(channel != 0)

    def visibility_channel(self, ownership_channel: np.ndarray) -> np.ndarray:
        """
        Returns a binary channel of visible cells from the perspective of the given player.
        """
        return maximum_filter(ownership_channel, size=3)

    def step(self, actions: dict[str, Action]) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Perform one step of the game
        """
        done_before_actions = self.is_done()
        # Process validity of moves, whether agents want to pass the turn,
        # and calculate intended amount of army to move (all available or split)
        moves = {}
        for agent, move in actions.items():
            pass_turn = move["pass"]
            if isinstance(move["cell"], np.ndarray):
                i = move["cell"][0]
                j = move["cell"][1]
            else:
                raise ValueError('Action key "cell" should be a numpy array.')
            direction = move["direction"]
            split_army = move["split"]
            # Skip if agent wants to pass the turn
            if pass_turn == 1:
                continue
            # Skip if the move is invalid
            if self.action_mask(agent)[i, j, direction] == 0:
                warnings.warn(
                    f"The submitted move byt agent {agent} does not take effect.\
                    Probably because you submitted an invalid move.",
                    UserWarning,
                )
                continue
            if split_army == 1:  # Agent wants to split the army
                army_to_move = self.channels.army[i, j] // 2
            else:  # Leave just one army in the source cell
                army_to_move = self.channels.army[i, j] - 1
            if army_to_move < 1:  # Skip if army size to move is less than 1
                continue
            moves[agent] = (i, j, direction, army_to_move)

        # Evaluate moves (smaller army movements are prioritized)
        for agent in sorted(moves, key=lambda x: moves[x][3]):
            si, sj, direction, army_to_move = moves[agent]

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, self.channels.army[si, sj] - 1)
            army_to_stay = self.channels.army[si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            if self.channels.ownership[agent][si, sj] == 0 or army_to_move < 1:
                continue

            di, dj = (
                si + DIRECTIONS[direction].value[0],
                sj + DIRECTIONS[direction].value[1],
            )  # destination indices

            # Figure out the target square owner and army size
            target_square_army = self.channels.army[di, dj]
            target_square_owner_idx = np.argmax(
                [self.channels.ownership[agent][di, dj] for agent in ["neutral"] + self.agents]
            )
            target_square_owner = (["neutral"] + self.agents)[target_square_owner_idx]
            if target_square_owner == agent:
                self.channels.army[di, dj] += army_to_move
                self.channels.army[si, sj] = army_to_stay
            else:
                # Calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - army_to_move)
                square_winner = agent if target_square_army < army_to_move else target_square_owner
                self.channels.army[di, dj] = remaining_army
                self.channels.army[si, sj] = army_to_stay
                self.channels.ownership[square_winner][di, dj] = 1
                if square_winner != target_square_owner:
                    self.channels.ownership[target_square_owner][di, dj] = 0

        if not done_before_actions:
            self.time += 1

        if self.is_done():
            # Give all cells of loser to winner
            winner = self.agents[0] if self.agent_won(self.agents[0]) else self.agents[1]
            loser = self.agents[1] if winner == self.agents[0] else self.agents[0]
            self.channels.ownership[winner] += self.channels.ownership[loser]
            self.channels.ownership[loser] = self.channels.passable * 0
        else:
            self._global_game_update()

        observations = self.get_all_observations()
        infos: dict[str, Any] = {agent: {} for agent in self.agents}
        return observations, infos

    def get_all_observations(self) -> dict[str, Observation]:
        """
        Returns observations for all agents.
        """
        return {agent: self.agent_observation(agent) for agent in self.agents}

    def _global_game_update(self) -> None:
        """
        Update game state globally.
        """

        owners = self.agents

        # every `increment_rate` steps, increase army size in each cell
        if self.time % self.increment_rate == 0:
            for owner in owners:
                self.channels.army += self.channels.ownership[owner]

        # Increment armies on general and city cells, but only if they are owned by player
        if self.time % 2 == 0 and self.time > 0:
            update_mask = self.channels.general + self.channels.city
            for owner in owners:
                self.channels.army += update_mask * self.channels.ownership[owner]

    def is_done(self) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        return any(self.agent_won(agent) for agent in self.agents)

    def get_infos(self) -> dict[str, Info]:
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        """
        players_stats = {}
        for agent in self.agents:
            army_size = np.sum(self.channels.army * self.channels.ownership[agent]).astype(int)
            land_size = np.sum(self.channels.ownership[agent]).astype(int)
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_winner": self.agent_won(agent),
            }
        return players_stats

    def agent_observation(self, agent: str) -> Observation:
        """
        Returns an observation for a given agent.
        """
        info = self.get_infos()
        opponent = self.agents[0] if agent == self.agents[1] else self.agents[1]
        visibility = self.visibility_channel(self.channels.ownership[agent])
        _observation = {
            "army": self.channels.army.astype(int) * visibility,
            "general": self.channels.general * visibility,
            "city": self.channels.city * visibility,
            "owned_cells": self.channels.ownership[agent] * visibility,
            "opponent_cells": self.channels.ownership[opponent] * visibility,
            "neutral_cells": self.channels.ownership_neutral * visibility,
            "visible_cells": visibility,
            "structure": self.channels.mountain + self.channels.city,
            "owned_land_count": info[agent]["land"],
            "owned_army_count": info[agent]["army"],
            "opponent_land_count": info[opponent]["land"],
            "opponent_army_count": info[opponent]["army"],
            "is_winner": int(info[agent]["is_winner"]),
            "timestep": self.time,
        }
        observation: Observation = {
            "observation": _observation,
            "action_mask": self.action_mask(agent),
        }

        return observation

    def agent_won(self, agent: str) -> bool:
        """
        Returns True if the agent won the game, False otherwise.
        """
        return all(
            self.channels.ownership[agent][general[0], general[1]] == 1 for general in self.general_positions.values()
        )
