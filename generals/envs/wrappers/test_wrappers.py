import gymnasium as gym
import numpy as np


class ActionMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super.__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space,
                "action_mask": gym.spaces.MultiBinary(env.action_space.n),
            }
        )

    def observation(self, observation):
        observation["action_mask"] = ...


class NormalizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(NormalizeObservationWrapper, self).__init__(env)
        grid_multi_binary = gym.spaces.MultiBinary(self.game.grid_dims)
        unit_box = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "army": gym.spaces.Box(
                            low=0, high=1, shape=self.game.grid_dims, dtype=np.float32
                        ),
                        "general": grid_multi_binary,
                        "city": grid_multi_binary,
                        "owned_cells": grid_multi_binary,
                        "opponent_cells": grid_multi_binary,
                        "neutral_cells": grid_multi_binary,
                        "visible_cells": grid_multi_binary,
                        "structure": grid_multi_binary,
                        "owned_land_count": unit_box,
                        "owned_army_count": unit_box,
                        "opponent_land_count": unit_box,
                        "opponent_army_count": unit_box,
                        "is_winner": gym.spaces.Discrete(2),
                        "timestep": unit_box,
                    }
                ),
                "action_mask": gym.spaces.MultiBinary(self.game.grid_dims + (4,)),
            }
        )

    def observation(self, observation):
        game = self.game
        _observation = (
            observation["observation"] if "observation" in observation else observation
        )
        _observation["army"] = _observation["army"] / game.max_army_value
        _observation["timestep"] = _observation["timestep"] / game.max_timestep
        _observation["owned_land_count"] = (
            _observation["owned_land_count"] / game.max_land_value
        )
        _observation["opponent_land_count"] = (
            _observation["opponent_land_count"] / game.max_land_value
        )
        _observation["owned_army_count"] = (
            _observation["owned_army_count"] / game.max_army_value
        )
        _observation["opponent_army_count"] = (
            _observation["opponent_army_count"] / game.max_army_value
        )
        observation["observation"] = _observation
        return observation


class RemoveActionMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(RemoveActionMaskWrapper, self).__init__(env)
        grid_multi_binary = gym.spaces.MultiBinary(self.game.grid_dims)
        unit_box = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "army": gym.spaces.Box(
                    low=0, high=1, shape=self.game.grid_dims, dtype=np.float32
                ),
                "general": grid_multi_binary,
                "city": grid_multi_binary,
                "owned_cells": grid_multi_binary,
                "opponent_cells": grid_multi_binary,
                "neutral_cells": grid_multi_binary,
                "visible_cells": grid_multi_binary,
                "structure": grid_multi_binary,
                "owned_land_count": unit_box,
                "owned_army_count": unit_box,
                "opponent_land_count": unit_box,
                "opponent_army_count": unit_box,
                "is_winner": gym.spaces.Discrete(2),
                "timestep": unit_box,
            }
        )

    def observation(self, observation):
        _observation = (
            observation["observation"] if "observation" in observation else observation
        )
        return _observation
