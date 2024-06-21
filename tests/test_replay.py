from generals.env import generals_v0
import generals.utils


map, action_sequence = generals.utils.load_replay("test")

env = generals_v0(map)
o, i = env.reset(seed=42, options={"replay": "test"})
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
