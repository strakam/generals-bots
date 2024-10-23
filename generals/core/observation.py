import numpy as np

from generals.core.game import DIRECTIONS, Game


def observation_from_simulator(game: Game, agent_id: str) -> "Observation":
    scores = {}
    for agent in game.agents:
        army_size = np.sum(game.channels.army * game.channels.ownership[agent]).astype(int)
        land_size = np.sum(game.channels.ownership[agent]).astype(int)
        scores[agent] = {
            "army": army_size,
            "land": land_size,
        }
    opponent = game.agents[0] if agent_id == game.agents[1] else game.agents[1]
    visible = game.channels.get_visibility(agent_id)
    invisible = 1 - visible
    army = game.channels.army.astype(int) * visible
    generals = game.channels.general * visible
    city = game.channels.city * visible
    owned_cells = game.channels.ownership[agent_id] * visible
    opponent_cells = game.channels.ownership[opponent] * visible
    neutral_cells = game.channels.ownership_neutral * visible
    visible_cells = visible
    structures_in_fog = invisible * (game.channels.mountain + game.channels.city)
    owned_land_count = scores[agent_id]["land"]
    owned_army_count = scores[agent_id]["army"]
    opponent_land_count = scores[opponent]["land"]
    opponent_army_count = scores[opponent]["army"]
    timestep = game.time

    return Observation(
        army=army,
        generals=generals,
        city=city,
        owned_cells=owned_cells,
        opponent_cells=opponent_cells,
        neutral_cells=neutral_cells,
        visible_cells=visible_cells,
        structures_in_fog=structures_in_fog,
        owned_land_count=owned_land_count,
        owned_army_count=owned_army_count,
        opponent_land_count=opponent_land_count,
        opponent_army_count=opponent_army_count,
        timestep=timestep,
    )


class Observation:
    def __init__(
        self,
        armies: np.ndarray,
        generals: np.ndarray,
        cities: np.ndarray,
        mountains: np.ndarray,
        owned_cells: np.ndarray,
        opponent_cells: np.ndarray,
        neutral_cells: np.ndarray,
        visible_cells: np.ndarray,
        structures_in_fog: np.ndarray,
        owned_land_count: int,
        owned_army_count: int,
        opponent_land_count: int,
        opponent_army_count: int,
        timestep: int,
    ):
        self.armies = armies
        self.generals = generals
        self.cities = cities
        self.mountains = mountains
        self.owned_cells = owned_cells
        self.opponent_cells = opponent_cells
        self.neutral_cells = neutral_cells
        self.visible_cells = visible_cells
        self.structures_in_fog = structures_in_fog
        self.owned_land_count = owned_land_count
        self.owned_army_count = owned_army_count
        self.opponent_land_count = opponent_land_count
        self.opponent_army_count = opponent_army_count
        self.timestep = timestep

    def action_mask(self) -> np.ndarray:
        """
        Function to compute valid actions from a given ownership mask.

        Valid action is an action that originates from agent's cell with atleast 2 units
        and does not bump into a mountain or fall out of the grid.
        Returns:
            np.ndarray: an NxNx4 array, where each channel is a boolean mask
            of valid actions (UP, DOWN, LEFT, RIGHT) for each cell in the grid.

            I.e. valid_action_mask[i, j, k] is 1 if action k is valid in cell (i, j).
        """
        height, width = self.owned_cells.shape

        ownership_channel = self.owned_cells
        more_than_1_army = (self.armies > 1) * ownership_channel
        owned_cells_indices = np.argwhere(more_than_1_army)
        valid_action_mask = np.zeros((height, width, 4), dtype=bool)

        if np.sum(ownership_channel) == 0:
            return valid_action_mask

        for channel_index, direction in enumerate(DIRECTIONS):
            destinations = owned_cells_indices + direction.value

            # check if destination is in grid bounds
            in_first_boundary = np.all(destinations >= 0, axis=1)
            in_height_boundary = destinations[:, 0] < height
            in_width_boundary = destinations[:, 1] < width
            destinations = destinations[in_first_boundary & in_height_boundary & in_width_boundary]

            # check if destination is road
            passable_cells = 1 - self.mountains
            # assert that every value is either 0 or 1 in passable cells
            assert np.all(np.isin(passable_cells, [0, 1])), f"{passable_cells}"
            passable_cell_indices = passable_cells[destinations[:, 0], destinations[:, 1]] == 1
            action_destinations = destinations[passable_cell_indices]

            # get valid action mask for a given direction
            valid_source_indices = action_destinations - direction.value
            valid_action_mask[valid_source_indices[:, 0], valid_source_indices[:, 1], channel_index] = 1.0

        return valid_action_mask

    def as_dict(self, with_mask=True):
        _obs = {
            "armies": self.armies,
            "generals": self.generals,
            "cities": self.cities,
            "mountains": self.mountains,
            "owned_cells": self.owned_cells,
            "opponent_cells": self.opponent_cells,
            "neutral_cells": self.neutral_cells,
            "visible_cells": self.visible_cells,
            "structures_in_fog": self.structures_in_fog,
            "owned_land_count": self.owned_land_count,
            "owned_army_count": self.owned_army_count,
            "opponent_land_count": self.opponent_land_count,
            "opponent_army_count": self.opponent_army_count,
            "timestep": self.timestep,
        }
        if with_mask:
            obs = {
                "observation": _obs,
                "action_mask": self.action_mask(),
            }
        else:
            obs = _obs
        return obs
