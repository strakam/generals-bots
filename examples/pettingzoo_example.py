import gymnasium as gym
from generals.agents import AgentFactory

# Initialize agents
random = AgentFactory.make_agent("random")
expander = AgentFactory.make_agent("expander")

agents = {
    random.id: random,
    expander.id: expander,
}  # Environment calls agents by name

# Create environment -- render modes: {None, "human"}
env = gym.make("pz-generals-v0", agent_ids=list(agents.keys()), render_mode="human")
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
