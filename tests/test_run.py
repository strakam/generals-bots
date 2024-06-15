import numpy as np
from generals.env import generals_v0
import generals.config as game_config

config = game_config.Config(
    grid_size=16,
    starting_positions=[[1, 1], [5, 5]],
    #    map_name='test_map'
)

env = generals_v0(config)
o, i, = env.reset(seed=42)

agent_names = ['red', 'blue']

while env.agents:
    actions = {}
    for agent in env.agents:
        mask = o[agent]['action_mask']
        valid_actions = np.argwhere(mask == 1)
        action = np.random.choice(len(valid_actions))
        actions[agent] = valid_actions[action]
    o, r, te, tr, i = env.step(actions)
    env.render()
env.close()

