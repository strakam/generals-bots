#  type: ignore
import gymnasium as gym
import numpy as np

from generals.envs import GymnasiumGenerals
from generals import GridFactory

agent_names = ["007", "Generalissimo"]

grid_factory = GridFactory(
    min_grid_dims=(24, 24),
    max_grid_dims=(24, 24),
)

# Create n_envs parallel environments
n_envs = 1
envs = gym.vector.AsyncVectorEnv(
    [lambda: GymnasiumGenerals(agents=agent_names, grid_factory=grid_factory, truncation=500) for _ in range(n_envs)],
)


# Observations have shape: (n_envs, n_agents, grid_height, grid_width, n_channels)
# To access observation of the first agent, you can do: observations[:, 0, :, :, :]

observations, infos = envs.reset()
terminated = [False] * len(observations)
truncated = [False] * len(observations)

while True:
    # Sample random actions for each agent
    agent_actions = [envs.single_action_space.sample() for _ in range(n_envs)]
    npc_actions = [envs.single_action_space.sample() for _ in range(n_envs)]

    # Stack actions together
    actions = np.stack([agent_actions, npc_actions], axis=1)
    observations, dummy_rewards, terminated, truncated, infos = envs.step(actions)

    # Extract masks for each agent
    masks = [np.stack([infos[agent_name]["masks"] for agent_name in agent_names])]

    # Since gymnasium support single agent (scalar) rewards, we pack them in 'infos',
    # where these rewards can be accessed like this:
    real_rewards = [
        np.stack([infos[agent_name]["reward"] for agent_name in agent_names])
    ]

    if any(terminated) or any(truncated):
        break
