import numpy as np
import gymnasium as gym
from . import config as conf
from typing import Tuple, Dict, List, Union
import importlib.resources

from scipy.ndimage import maximum_filter

class Game():
    def __init__(self, config: conf.Config, agents: List[str]):
        self.config = config
        self.agents = agents
        self.agent_id = {agent: i for i, agent in enumerate(agents)}
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
            for agent, general in zip(self.agents, self.config.starting_positions):
                map[general[0], general[1]] = self.agent_id[agent] + config.GENERAL

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
            'ownership_plain': ((map == config.PASSABLE) | (map == config.CITY)).astype(np.float32),
            **{f'ownership_{agent}': np.where(map == config.GENERAL+id, 1, 0).astype(np.float32) 
                for id, agent in enumerate(self.agents)}
        }

        # Public statistics about players
        self.player_stats = {agent: {'army': 1, 'land': 1} for agent in self.agents}

        # initialize city costs 
        self.city_costs = np.random.choice(range(11), size=spatial_dim).astype(np.float32) + 40
        self.channels['army'] += self.city_costs * self.channels['city']

        self.observation_space = gym.spaces.Dict({
            'army': gym.spaces.Box(low=0, high=np.inf, shape=spatial_dim, dtype=np.int32),
            'general': gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.int32),
            'city': gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.int32),
            'ownership': gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.int32),
            'opponent_ownership': gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.int32),
            'neutral_ownership': gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.int32),
            'mountain': gym.spaces.Box(low=0, high=1, shape=spatial_dim, dtype=np.int32)
        })

        self.action_space = gym.spaces.MultiDiscrete([self.grid_size, self.grid_size, 4])


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

    def action_mask(self, agent: str) -> np.ndarray:
        """
        Function to compute valid actions from a given ownership mask.

        Args:
            agent_id: str

        Returns:
            np.ndarray: an NxNx4 array, where for last channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.
        """

        ownership_channel = self.channels[f'ownership_{agent}']

        UP, DOWN, LEFT, RIGHT = self.config.UP, self.config.DOWN, self.config.LEFT, self.config.RIGHT
        owned_cells_indices = self.channel_to_indices(ownership_channel)
        valid_action_mask = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)

        for channel_index, direction in enumerate([UP, DOWN, LEFT, RIGHT]):
            destinations = owned_cells_indices + direction

            # check if destination is in grid bounds
            in_first_boundary = np.all(destinations >= 0, axis=1)
            in_second_boundary = np.all(destinations < self.grid_size, axis=1)
            destinations = destinations[in_first_boundary & in_second_boundary]

            # check if destination is road
            passable_cell_indices = self.channels['passable'][destinations[:, 0], destinations[:, 1]] == 1.
            action_destinations = destinations[passable_cell_indices]

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
    
    def step(self, actions: Dict[str, Tuple[Tuple[int, int], int]]) -> None:
        """
        Perform one step of the game

        Args:
            actions: dictionary of agent_id to action (this will be reworked)
        """
        # TODO -> update statistics
        # this is intended for 1v1 for now and might not be bug free
        directions = np.array([self.config.UP, self.config.DOWN, self.config.LEFT, self.config.RIGHT])
        agents = list(actions.keys())
        np.random.shuffle(agents) # random action order

        for agent in agents:

            source = actions[agent][:2] # x,y
            direction = actions[agent][2] # 0,1,2,3

            si, sj = source[0], source[1] # source indices
            di, dj = source[0] + directions[direction][0], source[1] + directions[direction][1] # destination indices

            # Only for parallel api test
            if not (0 <= di < self.grid_size and 0 <= dj < self.grid_size):
                continue

            moved_army_size = self.channels['army'][si, sj] - 1

            # check if the current player owns the source cell and has atleast 2 army size
            if moved_army_size <= 1 or self.channels[f'ownership_{agent}'][si, sj] == 0:
                continue

            target_square_army = self.channels['army'][di, dj]
            target_square_owner_idx = np.argmax(
                [self.channels[f'ownership_{agent}'][di, dj] for agent in self.agents]
            )
            target_square_owner = self.agents[target_square_owner_idx]
            
            if target_square_owner == agent:
                self.channels['army'][di, dj] += moved_army_size
                self.channels['army'][si, sj] = 1
            else:
                # calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - moved_army_size)
                winner = agent if target_square_army < moved_army_size else target_square_owner
                self.channels['army'][di, dj] = remaining_army
                self.channels['army'][si, sj] = 1
                self.channels[f'ownership_{winner}'][di, dj] = 1
                if winner != target_square_owner:
                    self.channels[f'ownership_{target_square_owner}'][di, dj] = 0

        self.time += 1
        self.global_game_update()

        observations = {agent: self.agent_observation(agent) for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminated, truncated, infos

    def global_game_update(self):
        """
        Update game state globally.
        """

        owners = ['plain'] + self.agents
        # every TICK_RATE steps, increase army size in each cell
        if self.time % self.config.increment_rate == 0:
            for owner in owners:
                self.channels['army'] += self.channels[f'ownership_{owner}']

        if self.time % 2 == 0 and self.time > 0:
            # increment armies on general and city cells, but only if they are owned by player
            update_mask = self.channels['general'] + self.channels['city']
            for owner in owners:
                self.channels['army'] += update_mask * self.channels[f'ownership_{owner}']
        

        # update player statistics
        for agent in self.agents:
            army_size = np.sum(self.channels['army'] * self.channels[f'ownership_{agent}']).astype(np.int32)
            land_size = np.sum(self.channels[f'ownership_{agent}']).astype(np.int32)
            self.player_stats[agent]['army'] = army_size
            self.player_stats[agent]['land'] = land_size

    def agent_observation(self, agent: str) -> Dict[str, Union[np.ndarray, List[Tuple[int, int]]]]:
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
        visibility = self.visibility_channel(self.channels[f'ownership_{agent}'])
        observation = {
            'army': self.channels['army'] * visibility,
            'general': self.channels['general'] * visibility,
            'city': self.channels['city'] * visibility,
            'ownership': self.channels[f'ownership_{agent}'] * visibility,
            'opponent_ownership': self.channels[f'ownership_{opponent}'] * visibility,
            'neutral_ownership': self.channels['ownership_plain'] * visibility,
            'mountain': self.channels['mountain'],
            'action_mask': self.action_mask(agent)
        }
        return observation

