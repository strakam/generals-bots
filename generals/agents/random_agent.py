import numpy as np

from generals.core.game import Action
from generals.core.observation import Observation

from .agent import Agent


class RandomAgent(Agent):
    def __init__(
        self,
        id: str = "Random",
        color: tuple[int, int, int] = (242, 61, 106),
        split_prob: float = 0.25,
        idle_prob: float = 0.05,
    ):
        super().__init__(id, color)

        self.idle_probability = idle_prob
        self.split_probability = split_prob

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """
        mask = observation["action_mask"]
        observation = observation["observation"]

        valid_actions = np.argwhere(mask == 1)
        if len(valid_actions) == 0:  # No valid actions
            return {
                "pass": 1,
                "cell": np.array([0, 0]),
                "direction": 0,
                "split": 0,
            }
        pass_turn = 0 if np.random.rand() > self.idle_probability else 1
        split_army = 0 if np.random.rand() > self.split_probability else 1

        action_index = np.random.choice(len(valid_actions))
        cell = valid_actions[action_index][:2]
        direction = valid_actions[action_index][2]

        action = {
            "pass": pass_turn,
            "cell": cell,
            "direction": direction,
            "split": split_army,
        }
        return action

    def reset(self):
        pass
