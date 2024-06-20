import numpy as np
from generals.env import generals_v0
import generals.config as game_config
import generals.utils

config = game_config.Config(
    grid_size=16,
    starting_positions=[[1, 1], [5, 5]],
    #    map_name='test_map'
)

map, action_sequence = generals.utils.load_replay("test")
env = generals_v0(config)
o, i = env.reset(seed=42, options={"map": map, "replay": True})
agent_names = ['red', 'blue']

index = 0
while env.agents:
    actions = {}
    for agent in env.agents:
        actions[agent] = action_sequence[index][agent]
    o, r, te, tr, i = env.step(actions)
    index += 1
    env.render()
env.close()
