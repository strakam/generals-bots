import numpy as np
from . import config as conf
from typing import Tuple, Dict, List, Union
import importlib.resources

from scipy.ndimage import maximum_filter

class Game():
    def __init__(self, config: conf.Config):
        self.config = config
        self.time = 0
        self.turn = 0

        # Create map layout
        spatial_dim = (self.config.grid_size, self.config.grid_size)

        if self.config.map_name:
            map = self.load_map(self.config.map_name)
            self.config.grid_size = map.shape[0]
        else:
            p_plain = 1 - self.config.mountain_density - self.config.town_density
            probs = [p_plain, self.config.mountain_density, self.config.town_density]
            map = np.random.choice([config.PASSABLE, config.MOUNTAIN, config.CITY], size=spatial_dim, p=probs)

            # Place generals
            for i, general in enumerate(self.config.starting_positions):
                map[general[0], general[1]] = i + config.GENERAL # TODO -> get real agent id 

        self.map = map
        self.grid_size = self.config.grid_size

        # Initialize channels
        # Army - army size in each cell
        # General - general mask (1 if general is in cell, 0 otherwise)
        # Mountain - mountain mask (1 if cell is mountain, 0 otherwise)
        # City - city mask (1 if cell is city, 0 otherwise)
        # Passable - passable mask (1 if cell is passable, 0 otherwise)
        # Ownership_i - ownership mask for player i (1 if player i owns cell, 0 otherwise)
        # Ownerhsip_0 - ownership mask for neutral cells that are passable (1 if cell is neutral, 0 otherwise)
        self.channels = {
            'army': np.where(map >= config.GENERAL, 1, 0).astype(np.float32),
            'general': np.where(map >= config.GENERAL, 1, 0).astype(np.float32),
            'mountain': np.where(map == config.MOUNTAIN, 1, 0).astype(np.float32),
            'city': np.where(map == config.CITY, 1, 0).astype(np.float32),
            'passable': (map != config.MOUNTAIN).astype(np.float32),
            'ownership_0': ((map == config.PASSABLE) | (map == config.CITY)).astype(np.float32),
            **{f'ownership_{i+1}': np.where(map == config.GENERAL+i, 1, 0).astype(np.float32) 
                for i in range(self.config.n_players)}
        }

        # Public statistics about players
        self.player_stats = {
            (i+1): {'army': 1, 'land': 1} for i in range(self.config.n_players)
        }

        # initialize city costs 
        self.city_costs = np.random.choice(range(11), size=spatial_dim).astype(np.float32) + 40
        self.channels['army'] += self.city_costs * self.channels['city']


    def load_map(self, map_name: str) -> np.ndarray:
        """
        Load map from file.

        Args:
            map_name: str

        TODO: should be moved to utils.py or somewhere 

        Returns:
            np.ndarray: map layout
        """
        try:
            with importlib.resources.path('generals.maps', map_name) as path:
                with open(path, 'r') as f:
                    map = np.array([list(line.strip()) for line in f]).astype(np.float32)
                return map
        except ValueError:
            raise ValueError('Invalid map format or shape')

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
            in_first_boundary = np.all(action_destinations >= 0, axis=1)
            in_second_boundary = np.all(action_destinations < self.grid_size, axis=1)
            action_destinations = action_destinations[in_first_boundary & in_second_boundary]

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
        agent_ids = list(actions.keys())
        np.random.shuffle(agent_ids) # random action order

        for agent_id in agent_ids:

            source = actions[agent_id][0]
            direction = actions[agent_id][1]

            si, sj = source[0], source[1] # source indices
            di, dj = source[0] + directions[direction][0], source[1] + directions[direction][1] # destination indices
            moved_army_size = self.channels['army'][si, sj] - 1

            # check if the current player owns the source cell and has atleast 2 army size
            if moved_army_size <= 1 or self.channels[f'ownership_{agent_id}'][si, sj] == 0:
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
                self.channels['army'][di, dj] = remaining_army
                self.channels['army'][si, sj] = 1
                self.channels[f'ownership_{winner}'][di, dj] = 1
                if winner != target_square_owner:
                    self.channels[f'ownership_{target_square_owner}'][di, dj] = 0

        self.time += 1
        self.global_game_update()

    def global_game_update(self):
        """
        Update game state globally.
        """

        # every TICK_RATE steps, increase army size in each cell
        if self.time % self.config.increment_rate == 0:
            for i in range(1, self.config.n_players + 1):
                self.channels['army'] += self.channels[f'ownership_{i}']

        if self.time % 2 == 0 and self.time > 0:
            # increment armies on general and city cells, but only if they are owned by player
            update_mask = self.channels['general'] + self.channels['city']
            for i in range(1, self.config.n_players + 1):
                self.channels['army'] += update_mask * self.channels[f'ownership_{i}']
        

        # update player statistics
        for i in range(1, self.config.n_players + 1):
            self.player_stats[i]['army'] = np.sum(self.channels['army'] * self.channels[f'ownership_{i}']).astype(np.int32)
            self.player_stats[i]['land'] = np.sum(self.channels[f'ownership_{i}']).astype(np.int32)

    def agent_observation(self, agent_id: int, view: str='channel') -> Dict[str, Union[np.ndarray, List[Tuple[int, int]]]]:
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
            agent_id: int

        Returns:
            np.ndarray: observation for the given agent
        """
        if view not in ['channel', 'list']:
            raise ValueError('view should be either channel or list')

        visibility = self.visibility_channel(self.channels[f'ownership_{agent_id}'])
        observation = {
            'army': self.channels['army'] * visibility,
            'general': self.channels['general'] * visibility,
            'city': self.channels['city'] * visibility,
            'ownership': self.channels[f'ownership_{agent_id}'] * visibility,
            'opponent_ownership': self.channels[f'ownership_{1-agent_id}'] * visibility,
            'neutral_ownership': self.channels['ownership_0'] * visibility,
            'mountain': self.channels['mountain']
        }
        if view == 'list':
            observation = {k: self.channel_to_indices(v) for k, v in observation.items()}
        return observation
        

        

