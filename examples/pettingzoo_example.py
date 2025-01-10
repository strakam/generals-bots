from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals

# Initialize agents
random = RandomAgent()
expander = ExpanderAgent()

# Names are used for the environment
agent_names = [random.id, expander.id]

# Store agents in a dictionary
agents = {
    random.id: random,
    expander.id: expander
}

# Create environment
env = PettingZooGenerals(agent_ids=agent_names, to_render=True)
observations, info = env.reset()

done = False
while not done:
    actions = {}
    for agent in env.agent_ids:
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = terminated or truncated
    env.render()
