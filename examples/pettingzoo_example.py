from generals.agents import AgentFactory
from generals.envs import PettingZooGenerals

# Initialize agents
random = AgentFactory.make_agent("random")
expander = AgentFactory.make_agent("expander")

agents = {
    random.id: random,
    expander.id: expander,
}
agent_ids = list(agents.keys()) # Environment calls agents by name

# Create environment
env = PettingZooGenerals(agents=agent_ids, render_mode="human")
observations, info = env.reset()

done = False
while not done:
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].act(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values()) or any(truncated.values())
    env.render()
