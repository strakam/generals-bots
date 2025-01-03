from generals.core.action import Action
from generals.core.observation import Observation
from generals.rewards.common import compute_num_generals_owned
from generals.rewards.reward_fn import RewardFn


class WinLoseRewardFn(RewardFn):
    """A simple reward function. +1 if the agent wins. -1 if they lose."""

    def __call__(self, prior_obs: Observation, prior_action: Action, obs: Observation) -> float:
        change_in_num_generals_owned = compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)

        return float(1 * change_in_num_generals_owned)
