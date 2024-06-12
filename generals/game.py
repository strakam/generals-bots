import numpy as np
from . import config as conf
from typing import Tuple, Dict, List, Union

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
        # Ownerhsip_0 - ownership mask for neutral cells (1 if cell is neutral, 0 otherwise)
        self.channels = {
            'army': np.where(map >= config.GENERAL, 1, 0).astype(np.float32),
            'general': np.where(map >= config.GENERAL, 1, 0).astype(np.float32),
            'mountain': np.where(map == config.MOUNTAIN, 1, 0).astype(np.float32),
            'city': np.where(map == config.CITY, 1, 0).astype(np.float32),
            'passable': (map == config.PASSABLE) | (map == config.CITY) | (map == config.GENERAL),
            'ownership_0': np.where(map == 0, 1, 0).astype(np.float32),
            **{f'ownership_{i+1}': np.where(map == config.GENERAL+i, 1, 0).astype(np.float32) 
                for i in range(self.config.n_players)}
        }

        self._action_buffer = []

    def valid_actions(self, agent_id: int, view: str='channel') -> Union[np.ndarray, List[Tuple[int, int]]]:
        """
        Function to compute valid actions from a given ownership mask.

        Args:
            agent_id: int

        Returns:
            np.ndarray: an NxNx4 array, where for last channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.
        """

        if view not in ['channel', 'list']:
            raise ValueError('view should be either channel or list')

        ownership_channel = self.channels[f'ownership_{agent_id}']

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

        output = valid_action_mask
        if view == 'list':
            output = [((x,y),z) for x,y,z in np.argwhere(valid_action_mask)]
        return output
            

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
    
    def step(self, actions: Dict[int, Tuple[Tuple[int, int], int]]) -> None:
        """
        Perform one step of the game

        Args:
            actions: dictionary of agent_id to action (this will be reworked)
        """
        # TODO -> update statistics
        # this is intended for 1v1 for now and might not be bug free
        directions = np.array([self.config.UP, self.config.DOWN, self.config.LEFT, self.config.RIGHT])
        for agent_id, (source, direction) in actions.items():
            si, sj = source[0], source[1]
            di, dj = source[0] + directions[direction][0], source[1] + directions[direction][1]
            moved_army_size = self.channels['army'][si, sj]
            if moved_army_size <= 1: # we have to move at least 1 army at former square
                continue
            target_square_army = self.channels['army'][di, dj]
            target_square_owner = np.argmax(
                [self.channels[f'ownership_{i}'][di, dj] for i in range(self.config.n_players + 1)]
            )
            
            if target_square_owner == agent_id:
                self.channels['army'][di, dj] += moved_army_size
                self.channels['army'][si, sj] = 1
            else:
                # calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - moved_army_size)
                winner = agent_id if target_square_army < moved_army_size else target_square_owner
                loser = agent_id if target_square_army > moved_army_size else target_square_owner
                self.channels['army'][di, dj] = remaining_army
                self.channels['army'][si, sj] = 1
                self.channels[f'ownership_{winner}'][di, dj] = 1.
                self.channels[f'ownership_{loser}'][di, dj] = 0.
                
        # every TICK_RATE steps, increase army size in each cell
        if self.time % self.config.tick_rate == 0:
            nonzero_army = np.nonzero(self.channels['army'])
            self.channels['army'][nonzero_army] += 1
        self.time += 1
