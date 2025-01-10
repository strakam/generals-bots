from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals

# Initialize agents
random = RandomAgent(id="Random")
expander = ExpanderAgent(id="Expander")

# Store agents in a dictionary
agents = {
    random.id: random,
    expander.id: expander
}

# Create environment
env = PettingZooGenerals(agent_ids=[random.id, expander.id], to_render=True)
observations, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    actions = {}
    for agent in env.agent_ids:
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()
