import time
from generals.env import generals_v0
import generals.utils


map, action_sequence = generals.utils.load_replay("test")

env = generals_v0(map)
o, i = env.reset(seed=42, options={"replay": "test"})
agent_names = ['red', 'blue']

index = 0
tick = 0
while env.agents:
    if not env.renderer.paused and tick % env.renderer.game_speed == 0:
        actions = {}
        for agent in env.agents:
            actions[agent] = action_sequence[index][agent]
        o, r, te, tr, i = env.step(actions)
        index += 1
    tick += 1
    env.render()
    time.sleep(0.064)
env.close()
