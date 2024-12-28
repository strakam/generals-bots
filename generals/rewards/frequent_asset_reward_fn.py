from generals.core.action import Action
from generals.core.observation import Observation
from generals.rewards.common import compute_num_cities_owned, compute_num_generals_owned, is_action_valid
from generals.rewards.reward_fn import RewardFn


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
