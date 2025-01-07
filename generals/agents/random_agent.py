import numpy as np

from generals.core.action import Action, compute_valid_move_mask
from generals.core.observation import Observation

from .agent import Agent


class RandomAgent(Agent):
    def __init__(
        self,
        id: str = "Random",
        split_prob: float = 0.25,
        idle_prob: float = 0.05,
    ):
        super().__init__(id)

        self.idle_probability = idle_prob
        self.split_probability = split_prob

    def act(self, observation: Observation) -> Action:
        """
        Randomly selects a valid action.
        """

        mask = compute_valid_move_mask(observation)

        # Skip the turn if there are no valid moves.
        valid_moves = np.argwhere(mask == 1)
        if len(valid_moves) == 0:
            return Action(to_pass=True)

        to_pass = 1 if np.random.rand() <= self.idle_probability else 0
        to_split = 1 if np.random.rand() <= self.split_probability else 0

        move_index = np.random.choice(len(valid_moves))
        (row, col) = valid_moves[move_index][:2]
        direction = valid_moves[move_index][2]

        action = Action(to_pass, row, col, direction, to_split)
        return action

    def reset(self):
        pass
