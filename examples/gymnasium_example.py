#  type: ignore
import gymnasium as gym
import numpy as np

from generals.envs import GymnasiumGenerals
from generals.agents import RandomAgent

agent_names = ["007", "Generalissimo"]

agents = {
    "007": RandomAgent(),
    "Generalissimo": RandomAgent(),
}

n_envs = 3
envs = gym.vector.AsyncVectorEnv(
    [lambda: GymnasiumGenerals(agent_ids=agent_names, truncation=500) for _ in range(n_envs)],
)


observations, infos = envs.reset()
terminated = [False] * len(observations)
truncated = [False] * len(observations)

while True:
    agent_1_obs = observations[:, 0, :, :] # Batch of observations for agent 1
    agent_2_obs = observations[:, 1, :, :] # Batch of observations for agent 2

    agent_actions = [[1,0,0,0,0] for _ in agent_1_obs] # Do nothing (placeholder)
    npc_actions = [[1,0,0,0,0] for _ in agent_2_obs] # Do nothing (placeholder)

    actions = np.stack([agent_actions, npc_actions], axis=1) # Stack actions together

    observations, rewards, terminated, truncated, infos = envs.step(actions)
    masks = [np.stack([info[-1] for info in infos[agent_name]]) for agent_name in agent_names]
    if any(terminated) or any(truncated):
        break
