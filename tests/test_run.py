from generals.env import generals_v0
import generals.config as game_config

config = game_config.Config(
    grid_size=10,
    starting_positions=[[1, 1], [5, 5]]
)

env = generals_v0(config)
env.reset(seed=42)


for t in range(1000000):
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
        action = env.action_space(agent).sample()
        env.step(action)
env.close()

