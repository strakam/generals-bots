import numpy as np
from .agent import Agent

from generals.config import DIRECTIONS


class ExpanderAgent(Agent):
    def __init__(self, name="Expander", color=(0, 130, 255)):
        self.name = name
        self.color = color

    def play(self, observation):
        """
        Heuristically selects a valid (expanding) action.
        Prioritizes capturing opponent and then neutral cells.
        """
        mask = observation["action_mask"]
        valid_actions = np.argwhere(mask == 1)
        if len(valid_actions) == 0:  # No valid actions
            return np.array([1, 0, 0, 0, 0])  # pass the move

        army = observation["army"]
        opponent = observation["opponent_cells"]
        neutral = observation["neutral_cells"]

        # Find actions that capture opponent or neutral cells
        actions_capture_opponent = np.zeros(len(valid_actions))
        actions_capture_neutral = np.zeros(len(valid_actions))
        for i, action in enumerate(valid_actions):
            di, dj = action[:-1] + DIRECTIONS[action[-1]]  # Destination cell indices
            if army[action[0], action[1]] <= army[di, dj] + 1:  # Can't capture
                continue
            elif opponent[di, dj]:
                actions_capture_opponent[i] = 1
            elif neutral[di, dj]:
                actions_capture_neutral[i] = 1

        if np.any(actions_capture_opponent):  # Capture random opponent cell if possible
            action_index = np.random.choice(np.nonzero(actions_capture_opponent)[0])
            action = valid_actions[action_index]
        elif np.any(actions_capture_neutral):  # Capture random neutral cell if possible
            action_index = np.random.choice(np.nonzero(actions_capture_neutral)[0])
            action = valid_actions[action_index]
        else:  # Otherwise, select a random valid action
            action_index = np.random.choice(len(valid_actions))
            action = valid_actions[action_index]

        # pass=[0] to indicate we want to move, split=[0] to indicate we want to move all troops
        action = np.concatenate(([0], action, [0]))

        return action

    def reset(self):
        pass
