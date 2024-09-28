import warnings
import numpy as np
import gymnasium as gym
from typing import Dict, List
from generals.config import DIRECTIONS, PASSABLE, MOUNTAIN, INCREMENT_RATE

from scipy.ndimage import maximum_filter


class Game:
    def __init__(self, map: np.ndarray, agents: List[str]):
        self.agents = agents
        self.time = 0

        self.grid_dims = (map.shape[0], map.shape[1])
        self.map = map

        self.general_positions = {
            agent: np.argwhere(map == chr(ord("A") + i))[0]
            for i, agent in enumerate(self.agents)
        }

        valid_generals = ["A", "B"]  # because generals are represented as letters

        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Mountain - mountain mask (1 if cell is mountain, 0 otherwise)
        # City - city mask (1 if cell is city, 0 otherwise)
        # Passable - passable mask (1 if cell is passable, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        # Ownerhsip_0 - ownership mask for neutral cells that are passable (1 if cell is neutral, 0 otherwise)
        # Initialize channels
        self.channels = {
            "army": np.where(np.isin(map, valid_generals), 1, 0).astype(np.float32),
            "general": np.where(np.isin(map, valid_generals), 1, 0).astype(bool),
            "mountain": np.where(map == MOUNTAIN, 1, 0).astype(bool),
            "city": np.where(np.char.isdigit(map), 1, 0).astype(bool),
            "passable": (map != MOUNTAIN).astype(bool),
            "ownership_neutral": ((map == PASSABLE) | (np.char.isdigit(map))).astype(
                bool
            ),
            **{
                f"ownership_{agent}": np.where(map == chr(ord("A") + id), 1, 0).astype(
                    bool
                )
                for id, agent in enumerate(self.agents)
            },
        }

        # City costs are 40 + digit in the cell
        base_cost = 40
        city_costs = np.where(np.char.isdigit(map), map, "0").astype(np.float32)
        self.channels["army"] += base_cost * self.channels["city"] + city_costs

        ##########
        # Spaces #
        ##########
        grid_multi_binary = gym.spaces.MultiBinary(self.grid_dims)
        self.observation_space = gym.spaces.Dict(
            {
                "army": gym.spaces.Box(
                    low=0, high=1e5, shape=self.grid_dims, dtype=np.float32
                ),
                "general": grid_multi_binary,
                "city": grid_multi_binary,
                "owned_cells": grid_multi_binary,
                "opponent_cells": grid_multi_binary,
                "neutral_cells": grid_multi_binary,
                "visibile_cells": grid_multi_binary,
                "structure": grid_multi_binary,
                "action_mask": gym.spaces.MultiBinary(
                    (self.grid_dims[0], self.grid_dims[1], 4)
                ),
                "owned_land_count": gym.spaces.Discrete(np.iinfo(np.int64).max),
                "owned_army_count": gym.spaces.Discrete(np.iinfo(np.int64).max),
                "opponent_land_count": gym.spaces.Discrete(np.iinfo(np.int64).max),
                "opponent_army_count": gym.spaces.Discrete(np.iinfo(np.int64).max),
                "is_winner": gym.spaces.MultiBinary(1),
                "timestep": gym.spaces.Discrete(np.iinfo(np.int64).max),
            }
        )

        self.action_space = gym.spaces.MultiDiscrete(
            [2, self.grid_dims[0], self.grid_dims[1], 4, 2]
        )

    def action_mask(self, agent: str) -> np.ndarray:
        """
        Function to compute valid actions from a given ownership mask.

        Valid action is an action that originates from agent's cell with atleast 2 units
        and does not bump into a mountain or fall out of the grid.

        Args:
            agent: str

        Returns:
            np.ndarray: an NxNx4 array, where each channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.

            I.e. valid_action_mask[i, j, k] is 1 if action k is valid in cell (i, j).
        """

        ownership_channel = self.channels[f"ownership_{agent}"]
        more_than_1_army = (self.channels["army"] > 1) * ownership_channel
        owned_cells_indices = self.channel_to_indices(more_than_1_army)
        valid_action_mask = np.zeros(
            (self.grid_dims[0], self.grid_dims[1], 4), dtype=bool
        )

        if self.is_done():
            return valid_action_mask

        for channel_index, direction in enumerate(DIRECTIONS):
            destinations = owned_cells_indices + direction

            # check if destination is in grid bounds
            in_first_boundary = np.all(destinations >= 0, axis=1)
            in_height_boundary = destinations[:, 0] < self.grid_dims[0]
            in_width_boundary = destinations[:, 1] < self.grid_dims[1]
            destinations = destinations[
                in_first_boundary & in_height_boundary & in_width_boundary
            ]

            # check if destination is road
            passable_cell_indices = (
                self.channels["passable"][destinations[:, 0], destinations[:, 1]] == 1.0
            )
            action_destinations = destinations[passable_cell_indices]

            # get valid action mask for a given direction
            valid_source_indices = action_destinations - direction
            valid_action_mask[
                valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index
            ] = 1.0
        return valid_action_mask

    def channel_to_indices(self, channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.

        Args:
            channel: one channel of the game grid

        Returns:
            np.ndarray: list of indices of cells with non-zero values.
        """
        return np.argwhere(channel != 0)

    def visibility_channel(self, ownership_channel: np.ndarray) -> np.ndarray:
        """
        Returns a binary channel of visible cells from the perspective of the given player.
        """
        return maximum_filter(ownership_channel, size=3)

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Perform one step of the game

        Args:
            actions: dictionary of agent name to action
        """
        done_before_actions = self.is_done()
        # Process validity of moves, whether agents want to pass the turn,
        # and calculate intended amount of army to move (all available or split)
        moves = {}
        for agent, move in actions.items():
            pass_turn, i, j, direction, split_army = move
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
                army_to_move = self.channels["army"][i, j] // 2
            else:  # Leave just one army in the source cell
                army_to_move = self.channels["army"][i, j] - 1
            if army_to_move < 1:  # Skip if army size to move is less than 1
                continue
            moves[agent] = (i, j, direction, army_to_move)

        # Evaluate moves (smaller army movements are prioritized)
        for agent in sorted(moves, key=lambda x: moves[x][3]):
            si, sj, direction, army_to_move = moves[agent]

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, self.channels["army"][si, sj] - 1)
            army_to_stay = self.channels["army"][si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            if self.channels[f"ownership_{agent}"][si, sj] == 0 or army_to_move < 1:
                continue

            di, dj = (
                si + DIRECTIONS[direction][0],
                sj + DIRECTIONS[direction][1],
            )  # destination indices

            # Figure out the target square owner and army size
            target_square_army = self.channels["army"][di, dj]
            target_square_owner_idx = np.argmax(
                [
                    self.channels[f"ownership_{agent}"][di, dj]
                    for agent in ["neutral"] + self.agents
                ]
            )
            target_square_owner = (["neutral"] + self.agents)[target_square_owner_idx]
            if target_square_owner == agent:
                self.channels["army"][di, dj] += army_to_move
                self.channels["army"][si, sj] = army_to_stay
            else:
                # Calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - army_to_move)
                square_winner = (
                    agent if target_square_army < army_to_move else target_square_owner
                )
                self.channels["army"][di, dj] = remaining_army
                self.channels["army"][si, sj] = army_to_stay
                self.channels[f"ownership_{square_winner}"][di, dj] = 1
                if square_winner != target_square_owner:
                    self.channels[f"ownership_{target_square_owner}"][di, dj] = 0

        if not done_before_actions:
            self.time += 1

        if self.is_done():
            # Give all cells of loser to winner
            winner = (
                self.agents[0] if self.agent_won(self.agents[0]) else self.agents[1]
            )
            loser = self.agents[1] if winner == self.agents[0] else self.agents[0]
            self.channels[f"ownership_{winner}"] += self.channels[f"ownership_{loser}"]
            self.channels[f"ownership_{loser}"] = self.channels["passable"] * 0
        else:
            self._global_game_update()

        observations = {agent: self._agent_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def get_all_observations(self):
        """
        Returns observations for all agents.
        """
        return {agent: self._agent_observation(agent) for agent in self.agents}

    def _global_game_update(self):
        """
        Update game state globally.
        """

        owners = self.agents

        # every TICK_RATE steps, increase army size in each cell
        if self.time % INCREMENT_RATE == 0:
            for owner in owners:
                self.channels["army"] += self.channels[f"ownership_{owner}"]

        # Increment armies on general and city cells, but only if they are owned by player
        if self.time % 2 == 0 and self.time > 0:
            update_mask = self.channels["general"] + self.channels["city"]
            for owner in owners:
                self.channels["army"] += (
                    update_mask * self.channels[f"ownership_{owner}"]
                )

    def is_done(self):
        """
        Returns True if the game is over, False otherwise.
        """
        return any(self.agent_won(agent) for agent in self.agents)

    def get_infos(self):
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        """
        players_stats = {}
        for agent in self.agents:
            army_size = np.sum(
                self.channels["army"] * self.channels[f"ownership_{agent}"]
            ).astype(np.int64)
            land_size = np.sum(self.channels[f"ownership_{agent}"]).astype(np.int64)
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_winner": self.agent_won(agent),
            }
        return players_stats

    def _agent_observation(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Returns an observation for a given agent.
        Args:
            agent: str
        """
        info = self.get_infos()
        opponent = self.agents[0] if agent == self.agents[1] else self.agents[1]
        visibility = self.visibility_channel(self.channels[f"ownership_{agent}"])
        observation = {
            "army": self.channels["army"] * visibility,
            "general": self.channels["general"] * visibility,
            "city": self.channels["city"] * visibility,
            "owned_cells": self.channels[f"ownership_{agent}"] * visibility,
            "opponent_cells": self.channels[f"ownership_{opponent}"] * visibility,
            "neutral_cells": self.channels["ownership_neutral"] * visibility,
            "visibile_cells": visibility,
            "structure": self.channels["mountain"] + self.channels["city"],
            "action_mask": self.action_mask(agent),
            "owned_land_count": info[agent]["land"],
            "owned_army_count": info[agent]["army"],
            "opponent_land_count": info[opponent]["land"],
            "opponent_army_count": info[opponent]["army"],
            "is_winner": np.array([info[agent]["is_winner"]], dtype=np.bool),
            "timestep": self.time,
        }
        return observation

    def agent_won(self, agent: str) -> bool:
        """
        Returns True if the agent won the game, False otherwise.
        """
        return all(
            self.channels[f"ownership_{agent}"][general[0], general[1]] == 1
            for general in self.general_positions.values()
        )
