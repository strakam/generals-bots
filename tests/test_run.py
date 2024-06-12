import numpy as np
from generals.env import generals_v0
import generals.config as game_config

config = game_config.Config(
    grid_size=10,
    starting_positions=[[1, 1], [5, 5]],
    map_name='test_map'
)

env = generals_v0(config)
env.reset(seed=42)


# for t in range(1000000):
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
#
#         if termination or truncation:
#             valid_actions = env.action_space(agent)
#             action = valid_actions[0]
#         else:
#             # this is where you would insert your policy
#             valid_actions = env.action_space(agent)
#             action = valid_actions[0]
#         print(f"player {agent} action: {action}")
#         env.step(action)
while env.agents:
    actions = {}
    for agent in env.agents:
        valid_actions = env.action_space(agent)
        rand = np.random.randint(0, len(valid_actions))
        actions[agent] = valid_actions[rand]
    o, r, te, tr, i = env.step(actions)
env.close()

