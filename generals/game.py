import numpy as np
import pygame
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
        Returns a mask of valid actions for agent_id
        Agent can move from any controlled cell to any adjacent cell that is not mountain
        """
        owned_cells = self.channels['ownership_' + str(agent_id)]
        owned_cells_indices = self.channel_as_list(owned_cells)
        possible_destinations = []
        for direction in [UP, DOWN, LEFT, RIGHT]:
            action_destinations = owned_cells_indices + direction
            # check if destination is in grid bounds
            first_boundary = np.all(action_destinations >= 0, axis=1)
            second_boundary = np.all(action_destinations < self.grid_size, axis=1)
            action_destinations = action_destinations[first_boundary & second_boundary]
            # check if destination is road
            passable_cell_indices = self.channels['passable'][action_destinations[:, 0], action_destinations[:, 1]] == 1
            action_destinations = action_destinations[passable_cell_indices]

            possible_destinations.append(action_destinations)
        return np.concatenate(possible_destinations)
            

    def channel_as_list(self, channel: np.ndarray): # TODO type for return
        """
        Returns a list of indices of cells from specified channel
        """
        return np.argwhere(channel != 0)

    def list_representation_all(self): # TODO type for return
        """
        Returns a list of indices of cells for each channel
        """
        return {k: self.channel_as_list(v) for k, v in self.channels.items()}
    
    def visibility_channel(self, agent_id: int):
        """
        Returns a mask of visible cells for agent_id
        """
        return maximum_filter(self.channels['ownership_' + str(agent_id)], size=3)
    
    def step(self, actions: Dict[int, Tuple[int, int]]):
        """
        Perform one step of the game
        """
        self.time += 1

        # every TICK_RATE steps, increase army size in each cell
        if self.time % TICK_RATE == 0:
            nonzero_army = np.nonzero(self.channels['army'])
            self.channels['army'][nonzero_army] += 1

        

