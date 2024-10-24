from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np

from .channels import Channels
from .config import DIRECTIONS
from .grid import Grid
from .observation import Observation

# Type aliases
Action: TypeAlias = dict[str, int | np.ndarray]
Info: TypeAlias = dict[str, Any]


class Game:
    def __init__(self, grid: Grid, agents: list[str]):
        # Agents
        self.agents = agents

        # Grid
        _grid = grid.grid
        self.channels = Channels(_grid, self.agents)
        self.grid_dims = (_grid.shape[0], _grid.shape[1])
        self.general_positions = {
            agent: np.argwhere(_grid == chr(ord("A") + i))[0] for i, agent in enumerate(self.agents)
        }

        # Time stuff
        self.time = 0
        self.increment_rate = 50

        # Limits
        self.max_army_value = 10_000
        self.max_land_value = np.prod(self.grid_dims)
        self.max_timestep = 100_000

        # Spaces
        grid_multi_binary = gym.spaces.MultiBinary(self.grid_dims)
        grid_discrete = np.ones(self.grid_dims, dtype=int) * self.max_army_value
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "armies": gym.spaces.MultiDiscrete(grid_discrete),
                        "generals": grid_multi_binary,
                        "cities": grid_multi_binary,
                        "mountains": grid_multi_binary,
                        "neutral_cells": grid_multi_binary,
                        "owned_cells": grid_multi_binary,
                        "opponent_cells": grid_multi_binary,
                        "fog_cells": grid_multi_binary,
                        "structures_in_fog": grid_multi_binary,
                        "owned_land_count": gym.spaces.Discrete(self.max_army_value),
                        "owned_army_count": gym.spaces.Discrete(self.max_army_value),
                        "opponent_land_count": gym.spaces.Discrete(self.max_army_value),
                        "opponent_army_count": gym.spaces.Discrete(self.max_army_value),
                        "timestep": gym.spaces.Discrete(self.max_timestep),
                    }
                ),
                "action_mask": gym.spaces.MultiBinary(self.grid_dims + (4,)),
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "pass": gym.spaces.Discrete(2),
                "cell": gym.spaces.MultiDiscrete(list(self.grid_dims)),
                "direction": gym.spaces.Discrete(4),
                "split": gym.spaces.Discrete(2),
            }
        )

    def step(self, actions: dict[str, Action]) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Perform one step of the game
        """
        done_before_actions = self.is_done()
        # Process validity of moves, whether agents want to pass the turn,
        # and calculate intended amount of army to move (all available or split)
        moves = {}
        for agent, move in actions.items():
            pass_turn = move["pass"]
            if isinstance(move["cell"], np.ndarray):
                i = move["cell"][0]
                j = move["cell"][1]
            else:
                raise ValueError('Action key "cell" should be a numpy array.')
            direction = move["direction"]
            split_army = move["split"]
            # Skip if agent wants to pass the turn
            if pass_turn == 1:
                continue
            if split_army == 1:  # Agent wants to split the army
                army_to_move = self.channels.armies[i, j] // 2
            else:  # Leave just one army in the source cell
                army_to_move = self.channels.armies[i, j] - 1
            if army_to_move < 1:  # Skip if army size to move is less than 1
                continue
            moves[agent] = (i, j, direction, army_to_move)

        # Evaluate moves (smaller army movements are prioritized)
        for agent in sorted(moves, key=lambda x: moves[x][3]):
            si, sj, direction, army_to_move = moves[agent]

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, self.channels.armies[si, sj] - 1)
            army_to_stay = self.channels.armies[si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            if self.channels.ownership[agent][si, sj] == 0 or army_to_move < 1:
                continue

            di, dj = (
                si + DIRECTIONS[direction].value[0],
                sj + DIRECTIONS[direction].value[1],
            )  # destination indices

            # Figure out the target square owner and army size
            target_square_army = self.channels.armies[di, dj]
            target_square_owner_idx = np.argmax(
                [self.channels.ownership[agent][di, dj] for agent in ["neutral"] + self.agents]
            )
            target_square_owner = (["neutral"] + self.agents)[target_square_owner_idx]
            if target_square_owner == agent:
                self.channels.armies[di, dj] += army_to_move
                self.channels.armies[si, sj] = army_to_stay
            else:
                # Calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - army_to_move)
                square_winner = agent if target_square_army < army_to_move else target_square_owner
                self.channels.armies[di, dj] = remaining_army
                self.channels.armies[si, sj] = army_to_stay
                self.channels.ownership[square_winner][di, dj] = 1
                if square_winner != target_square_owner:
                    self.channels.ownership[target_square_owner][di, dj] = 0

        if not done_before_actions:
            self.time += 1

        if self.is_done():
            # give all cells of loser to winner
            winner = self.agents[0] if self.agent_won(self.agents[0]) else self.agents[1]
            loser = self.agents[1] if winner == self.agents[0] else self.agents[0]
            self.channels.ownership[winner] += self.channels.ownership[loser]
            self.channels.ownership[loser] = self.channels.passable * 0
        else:
            self._global_game_update()

        observations = {agent: self.agent_observation(agent) for agent in self.agents}
        infos = self.get_infos()
        return observations, infos

    def _global_game_update(self) -> None:
        """
        Update game state globally.
        """

        owners = self.agents

        # every `increment_rate` steps, increase army size in each cell
        if self.time % self.increment_rate == 0:
            for owner in owners:
                self.channels.armies += self.channels.ownership[owner]

        # Increment armies on general and city cells, but only if they are owned by player
        if self.time % 2 == 0 and self.time > 0:
            update_mask = self.channels.generals + self.channels.cities
            for owner in owners:
                self.channels.armies += update_mask * self.channels.ownership[owner]

    def is_done(self) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        return any(self.agent_won(agent) for agent in self.agents)

    def get_infos(self) -> dict[str, Info]:
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        """
        players_stats = {}
        for agent in self.agents:
            army_size = np.sum(self.channels.armies * self.channels.ownership[agent]).astype(int)
            land_size = np.sum(self.channels.ownership[agent]).astype(int)
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_winner": self.agent_won(agent),
            }
        return players_stats

    def agent_observation(self, agent: str) -> Observation:
        """
        Returns an observation for a given agent.
        """
        scores = {}
        for _agent in self.agents:
            army_size = np.sum(self.channels.armies * self.channels.ownership[_agent]).astype(int)
            land_size = np.sum(self.channels.ownership[_agent]).astype(int)
            scores[_agent] = {
                "army": army_size,
                "land": land_size,
            }

        visible = self.channels.get_visibility(agent)
        invisible = 1 - visible

        opponent = self.agents[0] if agent == self.agents[1] else self.agents[1]

        armies = self.channels.armies.astype(int) * visible
        mountains = self.channels.mountains * visible
        generals = self.channels.generals * visible
        cities = self.channels.cities * visible
        neutral_cells = self.channels.ownership_neutral * visible
        owned_cells = self.channels.ownership[agent] * visible
        opponent_cells = self.channels.ownership[opponent] * visible
        structures_in_fog = invisible * (self.channels.mountains + self.channels.cities)
        fog_cells = invisible - structures_in_fog
        owned_land_count = scores[agent]["land"]
        owned_army_count = scores[agent]["army"]
        opponent_land_count = scores[opponent]["land"]
        opponent_army_count = scores[opponent]["army"]
        timestep = self.time

        return Observation(
            armies=armies,
            generals=generals,
            cities=cities,
            mountains=mountains,
            neutral_cells=neutral_cells,
            owned_cells=owned_cells,
            opponent_cells=opponent_cells,
            fog_cells=fog_cells,
            structures_in_fog=structures_in_fog,
            owned_land_count=owned_land_count,
            owned_army_count=owned_army_count,
            opponent_land_count=opponent_land_count,
            opponent_army_count=opponent_army_count,
            timestep=timestep,
        )

    def agent_won(self, agent: str) -> bool:
        """
        Returns True if the agent won the game, False otherwise.
        """
        return all(
            self.channels.ownership[agent][general[0], general[1]] == 1 for general in self.general_positions.values()
        )
