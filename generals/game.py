import numpy as np
import pygame # TODO: check if this is should be elsewhere
from . import game_config
from typing import Tuple, Dict

from scipy.ndimage import maximum_filter

PASSABLE = 0
MOUNTAIN = 1
CITY = 2
GENERAL = 3
ARMY = 4
OWNERSHIP = 5

GRID_SIZE = 10
SQUARE_SIZE = 40
WINDOW_HEIGHT = SQUARE_SIZE * GRID_SIZE + 50
WINDOW_WIDTH = SQUARE_SIZE * GRID_SIZE
VISUAL_OFFSET = 5

TICK_RATE = 10

UP = [-1, 0]
DOWN = [1, 0]
LEFT = [0, -1]
RIGHT = [0, 1]

COLORS = {
    "fog_of_war": (80, 83, 86),
    "player_1": (255, 0, 0),
    "player_2": (0, 0, 255),
}

class Game():
    def __init__(self, config: game_config.GameConfig):
        pygame.init()
        self.config = config
        self.grid_size = config.grid_size
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.time = 0

        # Create map layout
        spatial_dim = (self.config.grid_size, self.config.grid_size)

        p_plain = 1 - self.config.mountain_density - self.config.town_density
        probs = [p_plain, self.config.mountain_density, self.config.town_density]
        map = np.random.choice([PASSABLE, MOUNTAIN, CITY], size=spatial_dim, p=probs)
        self.map = map

        # Place generals
        for i, general in enumerate(self.config.starting_positions):
            map[general[0], general[1]] = i + GENERAL # TODO -> get real agent id 

        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Mountain - mountain mask (1 if cell is mountain, 0 otherwise)
        # City - city mask (1 if cell is city, 0 otherwise)
        # Passable - passable mask (1 if cell is passable, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        self.channels = {
            'army': np.where(map >= GENERAL, 1, 0).astype(np.float32),
            'general': np.where(map >= GENERAL, 1, 0).astype(np.float32),
            'mountain': np.where(map == MOUNTAIN, 1, 0).astype(np.float32),
            'city': np.where(map == CITY, 1, 0).astype(np.float32),
            'passable': (map == PASSABLE) | (map == CITY) | (map == GENERAL),
            **{f'ownership_{i+1}': np.where(map == GENERAL+i, 1, 0).astype(np.float32) 
                for i in range(self.config.n_players)}
        }

    def valid_actions(self, agent_id: int) -> np.ndarray:
        """
        Function to compute valid actions for a given agent.

        Args:
            agent_id: int

        Returns:
            np.ndarray: an NxNx4 array, where for last channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.
        """
        owned_cells = self.channels['ownership_' + str(agent_id)]
        owned_cells_indices = self.channel_to_indices(owned_cells)
        valid_action_mask = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.bool) # bool? 
        for channel_index, direction in enumerate([UP, DOWN, LEFT, RIGHT]):
            action_destinations = owned_cells_indices + direction

            # check if destination is in grid bounds
            first_boundary = np.all(action_destinations >= 0, axis=1)
            second_boundary = np.all(action_destinations < self.grid_size, axis=1)
            action_destinations = action_destinations[first_boundary & second_boundary]

            # check if destination is road
            passable_cell_indices = self.channels['passable'][action_destinations[:, 0], action_destinations[:, 1]] == 1
            action_destinations = action_destinations[passable_cell_indices]

            valid_source_indices = action_destinations - direction
            valid_action_mask[valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index] = True

        return valid_action_mask
            

    def channel_to_indices(self, channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells from specified a channel.

        Expected channels are ownerhsip, general, city, mountain.
        
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
        if self.time % TICK_RATE == 0:
            nonzero_army = np.nonzero(self.channels['army'])
            self.channels['army'][nonzero_army] += 1

        

