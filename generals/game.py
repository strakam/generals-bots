import numpy as np
import gymnasium as gym
from typing import Dict, List  # type: ignore
from .constants import PASSABLE, MOUNTAIN, GENERAL  # type: ignore
from .constants import UP, DOWN, LEFT, RIGHT  # type: ignore
from .constants import INCREMENT_RATE  # type: ignore

from scipy.ndimage import maximum_filter


class Game:
    def __init__(self, map: np.ndarray, agents: List[str]):
        self.agents = agents
        self.agent_id = {agent: i for i, agent in enumerate(agents)}
        self.time = 0

        spatial_dim = (map.shape[0], map.shape[1])
        self.map = map
        self.grid_size = spatial_dim[0]  # works only for square maps now

        self.general_positions = {
            agent: np.argwhere(map == chr(ord(GENERAL) + self.agent_id[agent]))[0]
            for agent in self.agents
        }

        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Mountain - mountain mask (1 if cell is mountain, 0 otherwise)
        # City - city mask (1 if cell is city, 0 otherwise)
        # Passable - passable mask (1 if cell is passable, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        # Ownerhsip_0 - ownership mask for neutral cells that are passable (1 if cell is neutral, 0 otherwise)
        # np where to use if map is in 'A-Z' 
        valid_generals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.channels = {
            "army": np.where(np.isin(map, valid_generals), 1, 0).astype(np.float32),
            "general": np.where(np.isin(map, valid_generals), 1, 0).astype(bool),
            "mountain": np.where(map == MOUNTAIN, 1, 0).astype(bool),
            "city": np.where(np.char.isdigit(map), 1, 0).astype(bool),
            "passable": (map != MOUNTAIN).astype(bool),
            "ownership_neutral": ((map == PASSABLE) | (np.char.isdigit(map))).astype(bool),
            **{
                f"ownership_{agent}": np.where(map == chr(ord(GENERAL) + id), 1, 0).astype(
                    bool
                )
                for id, agent in enumerate(self.agents)
            },
        }

        # initialize city costs (constant for now)
        base_cost = 40
        city_costs = np.where(np.char.isdigit(map), map, 0).astype(np.float32)
        self.channels["army"] += base_cost * self.channels["city"] + city_costs

        # Public statistics about players
        self.player_stats = {
            agent: {
                "army": 1,
                "land": 1,
                "general_position": self.general_positions[agent],
            }
            for agent in self.agents
        }

        box = gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "army": gym.spaces.Box(
                    low=0, high=np.inf, shape=spatial_dim, dtype=np.int32
                ),
                "general": box,
                "city": box,
                "ownership": box,
                "ownership_opponent": box,
                "ownership_neutral": box,
                "mountain": box,
            }
        )

        self.action_space = gym.spaces.MultiDiscrete(
            [self.grid_size, self.grid_size, 4]
        )

    def action_mask(self, agent: str) -> np.ndarray:
        """
        Function to compute valid actions from a given ownership mask.

        Args:
            agent_id: str

        Returns:
            np.ndarray: an NxNx4 array, where for last channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.
        """

        ownership_channel = self.channels[f"ownership_{agent}"]
        # if np.sum(ownership_channel) == 0:
        #     raise ValueError(f'Player {agent} has no cells')

        owned_cells_indices = self.channel_to_indices(ownership_channel)
        valid_action_mask = np.zeros(
            (self.grid_size, self.grid_size, 4), dtype=np.float32
        )

        for channel_index, direction in enumerate([UP, DOWN, LEFT, RIGHT]):
            destinations = owned_cells_indices + direction

            # check if destination is in grid bounds
            in_first_boundary = np.all(destinations >= 0, axis=1)
            in_second_boundary = np.all(destinations < self.grid_size, axis=1)
            destinations = destinations[in_first_boundary & in_second_boundary]

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
        Returns a list of indices of cells from specified a channel.

        Expected channels are ownership, general, city, mountain.

        Args:
            channel: one channel of the game grid

        Returns:
            np.ndarray: list of indices of cells with non-zero values.
        """
        return np.argwhere(channel != 0)

    def visibility_channel(self, ownership_channel: np.ndarray) -> np.ndarray:
        """
        Returns a binary channel of visible cells from the perspective of the given player.

        Args:
            agent_id: int
        """
        return maximum_filter(ownership_channel, size=3)

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Perform one step of the game

        Args:
            actions: dictionary of agent_id to action (this will be reworked)
        """
        # this is intended for 1v1 for now and might not be bug free
        directions = np.array([UP, DOWN, LEFT, RIGHT])
        agents = list(actions.keys())
        np.random.shuffle(agents)  # random action order

        for agent in agents:
            source = actions[agent][:2]  # x,y
            direction = actions[agent][2]  # 0,1,2,3

            si, sj = source[0], source[1]  # source indices
            di, dj = (
                source[0] + directions[direction][0],
                source[1] + directions[direction][1],
            )  # destination indices

            moved_army_size = self.channels["army"][si, sj] - 1

            # check if the current player owns the source cell and has atleast 2 army size
            if moved_army_size < 1 or self.channels[f"ownership_{agent}"][si, sj] == 0:
                continue

            target_square_army = self.channels["army"][di, dj]
            target_square_owner_idx = np.argmax(
                [
                    self.channels[f"ownership_{agent}"][di, dj]
                    for agent in ["neutral"] + self.agents
                ]
            )
            target_square_owner = (["neutral"] + self.agents)[target_square_owner_idx]

            if target_square_owner == agent:
                self.channels["army"][di, dj] += moved_army_size
                self.channels["army"][si, sj] = 1
            else:
                # calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - moved_army_size)
                winner = (
                    agent
                    if target_square_army < moved_army_size
                    else target_square_owner
                )
                self.channels["army"][di, dj] = remaining_army
                self.channels["army"][si, sj] = 1
                self.channels[f"ownership_{winner}"][di, dj] = 1
                if winner != target_square_owner:
                    self.channels[f"ownership_{target_square_owner}"][di, dj] = 0

        self.time += 1
        self._global_game_update()

        observations = {agent: self._agent_observation(agent) for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        terminated = {agent: self._agent_terminated(agent) for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminated, truncated, infos

    def _global_game_update(self):
        """
        Update game state globally.
        """

        owners = self.agents
        # every TICK_RATE steps, increase army size in each cell
        if self.time % INCREMENT_RATE == 0:
            for owner in owners:
                self.channels["army"] += self.channels[f"ownership_{owner}"]

        if self.time % 2 == 0 and self.time > 0:
            # increment armies on general and city cells, but only if they are owned by player
            update_mask = self.channels["general"] + self.channels["city"]
            for owner in owners:
                self.channels["army"] += (
                    update_mask * self.channels[f"ownership_{owner}"]
                )

        # update player statistics
        for agent in self.agents:
            army_size = np.sum(
                self.channels["army"] * self.channels[f"ownership_{agent}"]
            ).astype(np.int32)
            land_size = np.sum(self.channels[f"ownership_{agent}"]).astype(np.int32)
            self.player_stats[agent]["army"] = army_size
            self.player_stats[agent]["land"] = land_size

    def _agent_observation(self, agent: str) -> Dict[str, np.ndarray]:
        """
        Returns an observation for a given agent.
        The order of channels is as follows:
        - (visible) army
        - (visible) general
        - (visible) city
        - (visible) agent ownership
        - (visible) opponent ownership
        - (visible) neutral ownership
        - mountain

        !!! Currently supports only 1v1 games !!!

        Args:
            agent_id: int, currently only 1 or 2

        Returns:
            np.ndarray: observation for the given agent
        """
        opponent = self.agents[0] if agent == self.agents[1] else self.agents[1]
        visibility = self.visibility_channel(self.channels[f"ownership_{agent}"])
        observation = {
            "army": self.channels["army"] * visibility,
            "general": self.channels["general"] * visibility,
            "city": self.channels["city"] * visibility,
            "ownership": self.channels[f"ownership_{agent}"] * visibility,
            "ownership_opponent": self.channels[f"ownership_{opponent}"] * visibility,
            "ownership_neutral": self.channels["ownership_neutral"] * visibility,
            "mountain": self.channels["mountain"],
            "action_mask": self.action_mask(agent),
        }
        return observation

    def _agent_terminated(self, agent: str) -> bool:
        """
        Returns True if the agent is terminated, False otherwise.
        """
        general = self.general_positions[agent]
        return self.channels[f"ownership_{agent}"][general[0], general[1]] == 0
