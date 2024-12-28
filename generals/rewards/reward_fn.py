import abc

from generals.core.action import Action
from generals.core.observation import Observation


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
