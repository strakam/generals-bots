#  type: ignore
import gymnasium as gym
import numpy as np

from generals.envs import MultiAgentGymnasiumGenerals

agent_names = ["007", "Generalissimo"]

n_envs = 12
envs = gym.vector.AsyncVectorEnv(
    [lambda: MultiAgentGymnasiumGenerals(agents=agent_names, truncation=500) for _ in range(n_envs)],
)


observations, infos = envs.reset()
terminated = [False] * len(observations)
truncated = [False] * len(observations)

while True:
    agent_actions = [envs.single_action_space.sample() for _ in range(n_envs)]
    npc_actions = [envs.single_action_space.sample() for _ in range(n_envs)]

    # Stack actions together
    actions = np.stack([agent_actions, npc_actions], axis=1)
    observations, rewards, terminated, truncated, infos = envs.step(actions)
    masks = [np.stack([info[-1] for info in infos[agent_name]]) for agent_name in agent_names]
    if any(terminated) or any(truncated):
        print("DONE!")
        break
