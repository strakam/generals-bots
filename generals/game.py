import numpy as np
from . import config as conf
from typing import Tuple, Dict

from scipy.ndimage import maximum_filter

class Game():
    def __init__(self, config: conf.Config):
        self.config = config
        self.grid_size = config.grid_size
        self.time = 0

        # Create map layout
        spatial_dim = (self.config.grid_size, self.config.grid_size)

        p_plain = 1 - self.config.mountain_density - self.config.town_density
        probs = [p_plain, self.config.mountain_density, self.config.town_density]
        map = np.random.choice([config.PASSABLE, config.MOUNTAIN, config.CITY], size=spatial_dim, p=probs)
        self.map = map

        # Place generals
        for i, general in enumerate(self.config.starting_positions):
            map[general[0], general[1]] = i + config.GENERAL # TODO -> get real agent id 

        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Mountain - mountain mask (1 if cell is mountain, 0 otherwise)
        # City - city mask (1 if cell is city, 0 otherwise)
        # Passable - passable mask (1 if cell is passable, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        self.channels = {
            'army': np.where(map >= config.GENERAL, 1, 0).astype(np.float32),
            'general': np.where(map >= config.GENERAL, 1, 0).astype(np.float32),
            'mountain': np.where(map == config.MOUNTAIN, 1, 0).astype(np.float32),
            'city': np.where(map == config.CITY, 1, 0).astype(np.float32),
            'passable': (map == config.PASSABLE) | (map == config.CITY) | (map == config.GENERAL),
            **{f'ownership_{i+1}': np.where(map == config.GENERAL+i, 1, 0).astype(np.float32) 
                for i in range(self.config.n_players)}
        }

    def valid_actions(self, ownership_channel: np.ndarray) -> np.ndarray:
        """
        Function to compute valid actions from a given ownership mask.

        Args:
            agent_id: int

        Returns:
            np.ndarray: an NxNx4 array, where for last channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.
        """

        UP, DOWN, LEFT, RIGHT = self.config.UP, self.config.DOWN, self.config.LEFT, self.config.RIGHT
        owned_cells_indices = self.channel_to_indices(ownership_channel)
        valid_action_mask = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)

        for channel_index, direction in enumerate([UP, DOWN, LEFT, RIGHT]):
            action_destinations = owned_cells_indices + direction

            # check if destination is in grid bounds
            first_boundary = np.all(action_destinations >= 0, axis=1)
            second_boundary = np.all(action_destinations < self.grid_size, axis=1)
            action_destinations = action_destinations[first_boundary & second_boundary]

            # check if destination is road
            passable_cell_indices = self.channels['passable'][action_destinations[:, 0], action_destinations[:, 1]] == 1.
            action_destinations = action_destinations[passable_cell_indices]

            # get valid action mask for a given direction
            valid_source_indices = action_destinations - direction
            valid_action_mask[valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index] = 1.

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
    
    def step(self, actions: Dict[int, Tuple[int, int]]):
        """
        Perform one step of the game

        Args:
            actions: dictionary of agent_id to action (this will be reworked)
        """
        self.time += 1

        # every TICK_RATE steps, increase army size in each cell
        if self.time % self.config.tick_rate == 0:
            nonzero_army = np.nonzero(self.channels['army'])
            self.channels['army'][nonzero_army] += 1

        

