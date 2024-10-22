import gymnasium as gym
import numpy as np


class NormalizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        grid_multi_binary = gym.spaces.MultiBinary(self.game.grid_dims)
        unit_box = gym.spaces.Box(low=0, high=1, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "army": gym.spaces.Box(low=0, high=1, shape=self.game.grid_dims, dtype=np.float32),
                        "general": grid_multi_binary,
                        "city": grid_multi_binary,
                        "owned_cells": grid_multi_binary,
                        "opponent_cells": grid_multi_binary,
                        "neutral_cells": grid_multi_binary,
                        "visible_cells": grid_multi_binary,
                        "structures_in_fog": grid_multi_binary,
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
        _observation = observation["observation"] if "observation" in observation else observation
        _observation["army"] = np.array(_observation["army"] / game.max_army_value, dtype=np.float32)
        _observation["timestep"] = np.array([_observation["timestep"] / game.max_timestep], dtype=np.float32)
        _observation["owned_land_count"] = np.array(
            [_observation["owned_land_count"] / game.max_land_value], dtype=np.float32
        )
        _observation["opponent_land_count"] = np.array(
            [_observation["opponent_land_count"] / game.max_land_value],
            dtype=np.float32,
        )
        _observation["owned_army_count"] = np.array(
            [_observation["owned_army_count"] / game.max_army_value], dtype=np.float32
        )
        _observation["opponent_army_count"] = np.array(
            [_observation["opponent_army_count"] / game.max_army_value],
            dtype=np.float32,
        )
        observation["observation"] = _observation
        return observation


class RemoveActionMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        grid_multi_binary = gym.spaces.MultiBinary(self.game.grid_dims)
        unit_box = gym.spaces.Box(low=0, high=1, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "army": gym.spaces.Box(low=0, high=1, shape=self.game.grid_dims, dtype=np.float32),
                "general": grid_multi_binary,
                "city": grid_multi_binary,
                "owned_cells": grid_multi_binary,
                "opponent_cells": grid_multi_binary,
                "neutral_cells": grid_multi_binary,
                "visible_cells": grid_multi_binary,
                "structures_in_fog": grid_multi_binary,
                "owned_land_count": unit_box,
                "owned_army_count": unit_box,
                "opponent_land_count": unit_box,
                "opponent_army_count": unit_box,
                "is_winner": gym.spaces.Discrete(2),
                "timestep": unit_box,
            }
        )

    def observation(self, observation):
        _observation = observation["observation"] if "observation" in observation else observation
        return _observation


class ObservationAsImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.game.grid_dims + (14,), dtype=np.float32)

    def observation(self, observation):
        _observation = observation["observation"] if "observation" in observation else observation
        # broadcast owned_land_count and other unit_boxes to the shape of the grid
        _owned_land_count = np.broadcast_to(_observation["owned_land_count"], self.game.grid_dims)
        _owned_army_count = np.broadcast_to(_observation["owned_army_count"], self.game.grid_dims)
        _opponent_land_count = np.broadcast_to(_observation["opponent_land_count"], self.game.grid_dims)
        _opponent_army_count = np.broadcast_to(_observation["opponent_army_count"], self.game.grid_dims)
        _is_winner = np.broadcast_to(_observation["is_winner"], self.game.grid_dims)
        _timestep = np.broadcast_to(_observation["timestep"], self.game.grid_dims)
        _observation = np.stack(
            [
                _observation["army"],
                _observation["general"],
                _observation["city"],
                _observation["owned_cells"],
                _observation["opponent_cells"],
                _observation["neutral_cells"],
                _observation["visible_cells"],
                _observation["structures_in_fog"],
                _owned_land_count,
                _owned_army_count,
                _opponent_land_count,
                _opponent_army_count,
                _is_winner,
                _timestep,
            ],
            dtype=np.float32,
            axis=-1,
        )
        _observation = np.moveaxis(_observation, -1, 0)
        return _observation
