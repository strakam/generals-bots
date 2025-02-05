import abc

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


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        """
        It's common to see notation like r(s, a) in RL theory and it's accordingly common for environments
        to issue rewards based on the prior-observation & prior-action alone.

        We intentionally diverge from that pattern here by also expecting the current-observation
        for two reasons.

        The first reason is largely for convenience. Without the current-observation, we'd effectively
        have to mimic the game-logic by applying the prior-action to the prior-observation. That makes
        reward-functions more complex and adds computational cost, since the game does this update anyways.

        Second, even with prior-action the current-observation cannot be fully re-created, since we
        cannot know what the other agent will do. And, we may want to create reward functions that
        incorporate information about the opponent's prior-action, i.e. their action at time (t-1).

        Args:
            prior_obs: Observation of the prior state, i.e. at time (t-1).
            prior_action: Action taken at the prior time-step, i.e. at time (t-1).
            obs: Observation of the current state, i.e. at time t.

        Returns:
            reward: The reward provided at time-step t.
        """


class WinLoseRewardFn(RewardFn):
    """A simple reward function. +1 if the agent wins. -1 if they lose."""

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_num_generals_owned = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
        return float(1 * change_in_num_generals_owned)


class FrequentAssetRewardFn(RewardFn):
    """This reward function is fairly frequent -- every action/turn should generate some kind of reward. And
    it heavily takes into account the agent/players assets i.e. their land, army & cities.
    """

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_army_size = obs.owned_army_count - prior_obs.owned_army_count
        change_in_land_owned = obs.owned_land_count - prior_obs.owned_land_count
        change_in_num_cities_owned = compute_num_cities_owned(obs) - compute_num_cities_owned(prior_obs)
        change_in_num_generals_owned = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
        # Moderately reward valid actions & penalize invalid actions.
        valid_action_reward = 1 if is_action_valid(prior_action, prior_obs) else -5

        reward = (
            valid_action_reward
            + 1 * change_in_army_size
            + 5 * change_in_land_owned
            + 10 * change_in_num_cities_owned
            + 10_000 * change_in_num_generals_owned
        )

        return reward


class LandRewardFn(RewardFn):
    """A reward function focused on gaining territory. Provides positive reward for gaining land tiles."""

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_land_owned = obs.owned_land_count - prior_obs.owned_land_count
        return float(change_in_land_owned)
