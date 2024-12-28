"""
This module is for reuseable functions that aid in creating reward-functions.
For example, computing the number of cities owned by an agent based on an observation.
"""

from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation


def compute_num_cities_owned(observation: Observation) -> int:
    owned_cities_mask = observation.cities & observation.owned_cells
    num_cities_owned = owned_cities_mask.sum()
    return num_cities_owned


def compute_num_generals_owned(observation: Observation) -> int:
    owned_generals_mask = observation.generals & observation.owned_cells
    num_generals_owned = owned_generals_mask.sum()
    return num_generals_owned


def is_action_valid(action: Action, observation: Observation) -> bool:
    valid_move_mask = compute_valid_move_mask(observation)
    row, col, direction = action[1], action[2], action[3]

    # The actions' row & col may be out of bounds depending on
    # the agents implementation.
    if row >= valid_move_mask.shape[0] or col >= valid_move_mask.shape[1]:
        return False

    is_action_valid = valid_move_mask[row][col][direction]

    return is_action_valid
