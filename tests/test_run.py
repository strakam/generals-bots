import numpy as np
from generals.env import generals_v0
import generals.config as game_config

config = game_config.Config(
    grid_size=10,
    starting_positions=[[1, 1], [5, 5]],
    #    map_name='test_map'
)

env = generals_v0(config)
env.reset(seed=42)

agent_names = ['red', 'blue']

while env.agents:
    actions = {}
    for agent in env.agents:
        valid_actions = env.action_space(agent)
        action = valid_actions.sample()
        print(f'{agent} action: {action}')
        actions[agent] = action
    o, r, te, tr, i = env.step(actions)
    env.render()
env.close()

