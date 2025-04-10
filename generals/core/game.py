from typing import Any, TypeAlias

import numba as nb
import numpy as np

from .action import Action
from .channels import Channels
from .config import DIRECTIONS
from .grid import Grid
from .observation import Observation

# Type aliases

Info: TypeAlias = dict[str, Any]


@nb.njit(cache=True)
def calculate_army_size(armies, ownership):
    return np.int32(np.sum(armies * ownership))


@nb.njit(cache=True)
def calculate_land_size(ownership):
    return np.int32(np.sum(ownership))


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
        self.max_army_value = 100_000
        self.max_land_value = np.prod(self.grid_dims)
        self.max_timestep = 100_000

        self.winner = None
        self.loser = None

    def compute_agent_order(self, actions: dict[str, Action]) -> list[str]:
        """
        Compute the order of agents based on their actions with the following priority rules:
        1. Agent moving to a tile that another agent is leaving from has highest priority
        2. Agent moving to their own tile has second priority
        3. Larger army size being moved has third priority
        """
        # Extract action components for each agent
        moves = {}
        for agent, action in actions.items():
            pass_turn, si, sj, direction, _ = action
            if pass_turn == 1:
                return self.agents[:]

            # Calculate destination coordinates
            di = si + DIRECTIONS[direction].value[0]
            dj = sj + DIRECTIONS[direction].value[1]

            # Store source and destination for each agent
            moves[agent] = {
                "source": (si, sj),
                "dest": (di, dj),
                "army_size": self.channels.armies[si, sj] - 1,  # -1 because we always leave 1 behind
            }

        # Check first priority: moving to tile that other is leaving
        for agent in self.agents:
            dest = moves[agent]["dest"]
            for other_agent in self.agents:
                if agent != other_agent:
                    if dest == moves[other_agent]["source"]:
                        return [agent] + [a for a in self.agents if a != agent]

        # Check second priority: moving to own tile
        own_tile_movers = []
        other_tile_movers = []

        for agent in self.agents:
            di, dj = moves[agent]["dest"]
            # If agent is moving to their own tile, add to own_tile_movers
            if self.channels.ownership[agent][di, dj] == 1:
                own_tile_movers.append(agent)
            else:
                other_tile_movers.append(agent)

        # If any agent is moving to their own tile, prioritize them
        if own_tile_movers:
            return own_tile_movers + other_tile_movers

        # Third priority: army size being moved
        return sorted(self.agents, key=lambda x: moves[x]["army_size"], reverse=True)

    def step(self, actions: dict[str, Action]) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Perform one step of the game
        """
        done_before_actions = self.is_done()
        agent_order = self.compute_agent_order(actions)
        for agent in agent_order:
            pass_turn, si, sj, direction, split_army = actions[agent]

            # Skip if agent wants to pass the turn
            if pass_turn == 1:
                continue
            if split_army == 1:  # Agent wants to split the army
                army_to_move = self.channels.armies[si, sj] // 2
            else:  # Leave just one army in the source cell
                army_to_move = self.channels.armies[si, sj] - 1
            if army_to_move < 1:  # Skip if army size to move is less than 1
                continue

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, self.channels.armies[si, sj] - 1)
            army_to_stay = self.channels.armies[si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            still_owns_cell = self.channels.ownership[agent][si, sj] == 1
            has_more_than_one_army = army_to_move > 0
            if not still_owns_cell or not has_more_than_one_army:
                continue

            di, dj = (
                si + DIRECTIONS[direction].value[0],
                sj + DIRECTIONS[direction].value[1],
            )  # destination indices

            # Skip if the destination cell is not passable or out of bounds
            out_of_i = di < 0 or di >= self.grid_dims[0]
            out_of_j = dj < 0 or dj >= self.grid_dims[1]
            if out_of_i or out_of_j or self.channels.passable[di, dj] == 0:
                continue

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
                self.channels.ownership[square_winner][di, dj] = True
                if square_winner != target_square_owner:  # Agent captured new cell
                    self.channels.ownership[target_square_owner][di, dj] = False

                    # Here we want to check if the captured cell is the opponent's general
                    if target_square_owner in self.agents:
                        gi, gj = tuple(self.general_positions[target_square_owner])
                        if (di, dj) == (gi, gj):
                            self.loser = target_square_owner
                            self.winner = agent
                            break

        if not done_before_actions:
            self.time += 1

        if self.is_done():
            # give all cells of loser to winner
            winner = self.agents[0] if self.winner == self.agents[0] else self.agents[1]
            loser = self.agents[1] if winner == self.agents[0] else self.agents[0]
            self.channels.ownership[winner] += self.channels.ownership[loser]
            self.channels.ownership[loser] = np.full(self.grid_dims, False)
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
        return self.winner is not None

    def get_infos(self) -> dict[str, Info]:
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        - is_done: True if the game is over, False otherwise
        - is_winner: True if the player won, False otherwise
        """
        players_stats = {}
        for agent in self.agents:
            army_size = calculate_army_size(self.channels.armies, self.channels.ownership[agent])
            land_size = calculate_land_size(self.channels.ownership[agent])
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_done": self.is_done(),
                "is_winner": self.winner == agent,
            }
        return players_stats

    def agent_observation(self, agent: str) -> Observation:
        """
        Returns an observation for a given agent.
        """
        scores = {}
        for _agent in self.agents:
            army_size = calculate_army_size(self.channels.armies, self.channels.ownership[_agent])
            land_size = calculate_land_size(self.channels.ownership[_agent])
            scores[_agent] = {
                "army": army_size,
                "land": land_size,
            }

        visible = self.channels.get_visibility(agent)
        invisible = 1 - visible

        opponent = self.agents[0] if agent == self.agents[1] else self.agents[1]

        armies = self.channels.armies * visible
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
        priority = 0  # Backwards compatibility, but not used anymore

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
            priority=priority,
        )
