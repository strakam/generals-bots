import numpy as np
import pygame
from . import game_config
from typing import Tuple, Dict

from scipy.ndimage import maximum_filter

ROAD = 0
TERRAIN = 1
TOWN = 2
GENERAL = 3
ARMY = 4
OWNERSHIP = 5

GRID_SIZE = 10
SQUARE_SIZE = 40
WINDOW_SIZE = SQUARE_SIZE * GRID_SIZE
VISUAL_OFFSET = 5

TICK_RATE = 10

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
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()

        self.time = 0

        # Create map layout
        p_plain = 1 - self.config.terrain_density - self.config.town_density
        probs = [p_plain, self.config.terrain_density, self.config.town_density]
        map = np.random.choice([ROAD, TERRAIN, TOWN], size=(self.config.grid_size, self.config.grid_size), p=probs)

        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        spatial_dim = (self.config.grid_size, self.config.grid_size)
        self.channels = {
            'army': np.where(map == GENERAL, 1, 0).astype(np.float32),
            'road': np.where(map == ROAD, 1, 0).astype(np.float32),
            'general': np.zeros(spatial_dim, dtype=np.float32),
            **{f'ownership_{i+1}': np.zeros(spatial_dim, dtype=np.float32) for i in range(self.config.n_players)}
        }

        # make general squares passable and finish channel initialization
        for general in self.config.starting_positions:
            map[general[0], general[1]] = ROAD # general square is considered passable
            self.channels['general'][general[0], general[1]] = 1


        # Place terrain, castle, road to corresponding channels
        for i, channel_name in enumerate(['road', 'terrain', 'castle']):
            self.channels[channel_name] = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
            self.channels[channel_name][map == i] = 1

        # Place generals to 'general' channel
        for general in self.config.starting_positions:
            self.channels['general'][general[0], general[1]] = 1
            self.channels['army'][general[0], general[1]] = 1

        for i in range(self.config.n_players):
            x, y = self.config.starting_positions[i]
            self.channels[f'ownership_{i+1}'][x, y] = 1

    def channel_as_list(self, channel: np.ndarray) -> list[Tuple[int, int]]:
        """
        Returns a list of indices of cells from specified channel
        """
        return np.argwhere(channel != 0)

    def list_representation_all(self) -> Dict[str, list[Tuple[int, int]]]:
        """
        Returns a list of indices of cells for each channel
        """
        return {k: self.channel_as_list(v) for k, v in self.channels.items()}
    
    def visibility_mask(self, agent_id: int):
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

        

