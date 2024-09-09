import numpy as np

class Agent:
    def __init__(self, name):
        self.name = name

    def play(self):
        raise NotImplementedError

    def __str__(self):
        return self.name


class RandomAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def play(self, observation):
        mask = observation['action_mask']
        valid_actions = np.argwhere(mask == 1)
        action = np.random.choice(len(valid_actions))
        return valid_actions[action]
