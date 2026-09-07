"""Visualize a game between agents using the pygame GUI.

Two players by default; set TEAMS for a team game or a free-for-all, e.g.
TEAMS = [0, 0, 1, 1] (2v2) or TEAMS = [0, 1, 2, 3] (4-player FFA). One agent
is created per player; sight is shared within a team, so in 2v2 each player's
view (click a name in the scoreboard to toggle it) covers its teammate's land.
"""
import time

import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent
from generals.gui import ReplayGUI

# Configuration
GRID_DIMS = (10, 10)
TRUNCATION = 500
FPS = 10
TEAMS = None            # None = classic 1v1; try [0, 0, 1, 1] or [0, 1, 2, 3]

teams = [0, 1] if TEAMS is None else list(TEAMS)
num_players = len(teams)

# Create environment and agents (one per player)
env = GeneralsEnv(
    grid_dims=GRID_DIMS,
    truncation=TRUNCATION,
    teams=teams,
    max_generals_distance=4 if num_players == 2 else None,
    min_generals_distance=3,
)
agents = [RandomAgent(id="Random") if i % 2 == 0 else ExpanderAgent(id="Expander") for i in range(num_players)]
agent_ids = [f"{agent.id} P{i} (team {teams[i]})" for i, agent in enumerate(agents)]

# Initialize game
key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

# Create GUI
gui = ReplayGUI(state, agent_ids=agent_ids)

terminated = truncated = False
step_count = 0

while not (terminated or truncated):
    key, *subkeys = jrandom.split(key, num_players + 1)
    actions = jnp.stack([agents[i].act(get_observation(state, i), subkeys[i]) for i in range(num_players)])

    timestep, state = env.step(state, actions, pool)

    gui.update(state, timestep.info)
    gui.tick(fps=FPS)

    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    step_count += 1

winner_team = int(timestep.info.winner)
if winner_team >= 0:
    members = [agent_ids[i] for i in range(num_players) if teams[i] == winner_team]
    print(f"Game over after {step_count} steps! Winner: team {winner_team} ({', '.join(members)})")
else:
    print(f"Game over after {step_count} steps! Winner: None (draw)")

time.sleep(2)
gui.close()
