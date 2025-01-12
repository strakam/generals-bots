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
env = PettingZooGenerals(agents=agent_names, render_mode="human")
observations, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()
