from generals.env import generals_v0
from generals.agents import RandomAgent
from generals.config import GameConfig

agents = {
    "red": RandomAgent("red"),
    "blue": RandomAgent("blue")
}

agent_names = list(agents.keys())

game_config = GameConfig(
    grid_size=4,
    agent_names=agent_names
)

env = generals_v0(game_config, render_mode="none")
observations, info = env.reset()

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        actions[agent] = agents[agent].play(observations[agent])
    observations, rewards, terminated, truncated, info = env.step(actions)
