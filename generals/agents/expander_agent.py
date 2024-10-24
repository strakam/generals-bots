import numpy as np

from generals.core.config import Direction
from generals.core.game import Action
from generals.core.observation import Observation

from .agent import Agent


class ExpanderAgent(Agent):
    def __init__(self, id: str = "Expander", color: tuple[int, int, int] = (0, 130, 255)):
        super().__init__(id, color)

    def act(self, observation: Observation) -> Action:
        """
        Heuristically selects a valid (expanding) action.
        Prioritizes capturing opponent and then neutral cells.
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

        army = observation["armies"]
        opponent = observation["opponent_cells"]
        neutral = observation["neutral_cells"]

        # Find actions that capture opponent or neutral cells
        actions_capture_opponent = np.zeros(len(valid_actions))
        actions_capture_neutral = np.zeros(len(valid_actions))

        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        for i, action in enumerate(valid_actions):
            di, dj = action[:-1] + directions[action[-1]].value  # Destination cell indices
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

        action = {
            "pass": 0,
            "cell": action[:2],
            "direction": action[2],
            "split": 0,
        }
        return action

    def reset(self):
        pass
