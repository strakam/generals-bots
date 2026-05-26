"""Visualize a multi-agent game using the pygame GUI.

By default runs 4 Expander agents in a 1v1v1v1 free-for-all. Tweak NUM_PLAYERS,
TEAMS, and the agents list below to try other configurations.
"""
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent
from generals.gui import ReplayGUI

# Configuration
GRID_DIMS = (15, 15)
TRUNCATION = 1000
FPS = 8

NUM_PLAYERS = 4
TEAMS = jnp.arange(NUM_PLAYERS, dtype=jnp.int32)  # FFA. Use jnp.array([0,0,1,1]) for 2v2.

# One agent per player. Mix and match freely.
agents = [ExpanderAgent(id=f"Expander{i}") for i in range(NUM_PLAYERS)]

env = GeneralsEnv(
    grid_dims=GRID_DIMS,
    truncation=TRUNCATION,
    num_players=NUM_PLAYERS,
    teams=TEAMS,
    min_generals_distance=4,
)

key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

gui = ReplayGUI(state, agent_ids=[a.id for a in agents])

terminated = truncated = False
step_count = 0
while not (terminated or truncated):
    key, *subkeys = jrandom.split(key, NUM_PLAYERS + 1)
    actions = jnp.stack([
        agents[i].act(get_observation(state, i), subkeys[i])
        for i in range(NUM_PLAYERS)
    ])

    timestep, state = env.step(state, actions, pool)

    gui.update(state, timestep.info)
    gui.tick(fps=FPS)

    terminated = bool(timestep.terminated)
    truncated = bool(timestep.truncated)
    step_count += 1

winner_team = int(timestep.info.winner)
print(f"Game over after {step_count} steps. Winning team: {winner_team}")
print(f"Eliminated: {list(map(bool, timestep.last_state.eliminated))}")

import time
time.sleep(3)
gui.close()
