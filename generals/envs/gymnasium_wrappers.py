import gymnasium as gym
import numpy as np


class RemoveActionMaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]

    def observation(self, observation):
        _observation = observation["observation"] if "observation" in observation else observation
        return _observation


class ObservationAsImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n_obs_keys = len(self.observation_space["observation"].items())
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=0, high=1, shape=self.game.grid_dims + (n_obs_keys,), dtype=np.float32
                ),
                "action_mask": gym.spaces.MultiBinary(self.game.grid_dims + (4,)),
            }
        )

    def observation(self, observation):
        game = self.game
        _obs = observation["observation"] if "observation" in observation else observation
        _obs = (
            np.stack(
                [
                    _obs["armies"] / game.max_army_value,
                    _obs["generals"],
                    _obs["cities"],
                    _obs["mountains"],
                    _obs["neutral_cells"],
                    _obs["owned_cells"],
                    _obs["opponent_cells"],
                    _obs["fog_cells"],
                    _obs["structures_in_fog"],
                    np.ones(game.grid_dims) * _obs["owned_land_count"] / game.max_land_value,
                    np.ones(game.grid_dims) * _obs["owned_army_count"] / game.max_army_value,
                    np.ones(game.grid_dims) * _obs["opponent_land_count"] / game.max_land_value,
                    np.ones(game.grid_dims) * _obs["opponent_army_count"] / game.max_army_value,
                    np.ones(game.grid_dims) * _obs["timestep"] / game.max_timestep,
                    np.ones(game.grid_dims) * _obs["priority"],
                ]
            )
            .astype(np.float32)
            .transpose(1, 2, 0)
        )
        return _obs
