import numpy as np
import pygame
from . import grid
from . import game_config
from generals import utils
from typing import List, Tuple, Dict

ROAD = 0
TERRAIN = 1
TOWN = 2
GENERAL = 3
ARMY = 4
OWNERSHIP = 5

class Game():
    def __init__(self, config: game_config.GameConfig):
        pygame.init()
        self.config = config
        self.grid_size = config.grid_size
        self.screen = pygame.display.set_mode((utils.WINDOW_SIZE, utils.WINDOW_SIZE))
        self.clock = pygame.time.Clock()

        # Create map layout

        p_plain = 1 - self.config.terrain_density - self.config.town_density
        probs = [p_plain, self.config.terrain_density, self.config.town_density]
        map = np.random.choice([ROAD, TERRAIN, TOWN], size=(self.config.grid_size, self.config.grid_size), p=probs)

        # place generals
        for general in self.config.starting_positions:
            map[general[0], general[1]] = 0 # general square is considered passable, so we give it 0
        
        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        spatial_dim = (self.config.grid_size, self.config.grid_size)
        self.channels = {
            'army': np.zeros(spatial_dim, dtype=np.float32),
            'general': np.zeros(spatial_dim, dtype=np.float32),
            **{f'ownership_{i}': np.zeros(spatial_dim, dtype=np.float32) for i in range(self.config.n_players)}
        }

        # Place terrain, castle, road to corresponding channels
        for i, channel_name in enumerate(['road', 'terrain', 'castle']):
            self.channels[channel_name] = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
            self.channels[channel_name][map == i] = 1

        # Place generals to 'general' channel
        for general in self.config.starting_positions:
            self.channels['general'][general[0], general[1]] = 1

    def render(self):
        utils.draw_static_parts(self.screen, self.list_representation())
        pygame.display.flip()

    def list_representation(self) -> Dict[str, list[Tuple[int, int]]]:
        """
        Returns a list of indices of cells for each channel
        """
        return {channel_name: np.argwhere(channel == 1) for channel_name, channel in self.channels.items()}

