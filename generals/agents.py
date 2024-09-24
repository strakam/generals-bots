import numpy as np

from generals.config import DIRECTIONS

class Agent:
    """
    Base class for all agents.
    """
    def __init__(self, name):
        self.name = name

    def play(self):
        """
        This method should be implemented by the child class.
        It should receive an observation and return an action.
        """
        raise NotImplementedError

    def __str__(self):
        return self.name


class RandomAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def play(self, observation):
        """
        Randomly selects a valid action.
        """
        mask = observation['action_mask']
        valid_actions = np.argwhere(mask == 1)
        action_index = np.random.choice(len(valid_actions))
        action = np.array([1])
        action = np.append(action, valid_actions[action_index])
        # append 1 or 0 randomly to the action (to say whether to send half of troops or all troops)
        action = np.append(action, np.random.choice([0, 1]))
        return action

class ExpanderAgent(Agent):
    def __init__(self, name):
        super().__init__(name)

    def play(self, observation):
        """
        Heuristically selects a valid (expanding) action.
        Prioritizes capturing opponent and then neutral cells.
        """
        mask = observation["action_mask"]
        army = observation["army"]

        valid_actions = np.argwhere(mask == 1)
        actions_with_more_than_1_army = (
            army[valid_actions[:, 0], valid_actions[:, 1]] > 1
        )
        if np.sum(actions_with_more_than_1_army) == 0:
            return np.array([0, 0, 0, 0, 0])  # IDLE move

        valid_actions = valid_actions[actions_with_more_than_1_army]

        opponent = observation["ownership_opponent"]
        neutral = observation["ownership_neutral"]

        # find actions that capture opponent or neutral cells
        actions_to_opponent = np.zeros(len(valid_actions))
        actions_to_neutral = np.zeros(len(valid_actions))
        for i, action in enumerate(valid_actions):
            destination = action[:-1] + DIRECTIONS[action[-1]]
            if army[action[0], action[1]] <= army[destination[0], destination[1]] + 1:
                continue
            elif opponent[destination[0], destination[1]]:
                actions_to_opponent[i] = 1
            if neutral[destination[0], destination[1]]:
                actions_to_neutral[i] = 1

        actions_to_neutral_indices = np.argwhere(actions_to_neutral == 1).flatten()
        actions_to_opponent_indices = np.argwhere(actions_to_opponent == 1).flatten()
        if len(actions_to_opponent_indices) > 0:
            # pick random action that captures an opponent cell
            action_index = np.random.choice(len(actions_to_opponent_indices))
            action = valid_actions[actions_to_opponent_indices[action_index]]
        elif len(actions_to_neutral_indices) > 0:
            # or pick random action that captures a neutral cell
            action_index = np.random.choice(len(actions_to_neutral_indices))
            action = valid_actions[actions_to_neutral_indices[action_index]]
        else:  # otherwise pick a random action
            action_index = np.random.choice(len(valid_actions))
            action = valid_actions[action_index]

        # append 0 to the action (to send all available troops)
        final_action = np.array([1])
        final_action = np.append(final_action, action)
        final_action = np.append(final_action, 0)
        return final_action
