from .agent import Agent
import numpy as np


class RandomAgent(Agent):
    def __init__(
        self, name="Random", color=(255, 0, 0), split_prob=0.25, idle_prob=0.05
    ):
        self.name = name
        self.color = color

        self.idle_probability = idle_prob
        self.split_probability = split_prob

    def play(self, observation):
        """
        Randomly selects a valid action.
        """
        mask = observation["action_mask"]
        observation = observation["observation"]

        valid_actions = np.argwhere(mask == 1)
        if len(valid_actions) == 0:  # No valid actions
            return (1, (0, 0), 0, 0)

        pass_turn = 0 if np.random.rand() > self.idle_probability else 1
        split_army = 0 if np.random.rand() > self.split_probability else 1

        action_index = np.random.choice(len(valid_actions))
        cell = valid_actions[action_index][:2]
        direction = valid_actions[action_index][2]

        action = (pass_turn, cell, direction, split_army)
        return action

    def reset(self):
        pass
