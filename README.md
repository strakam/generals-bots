<div align="center">

![Gameplay GIF](https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/gifs/wider_gameplay.gif)

## **Generals.io Bots**

[![CodeQL](https://github.com/strakam/generals-bots/actions/workflows/codeql.yml/badge.svg)](https://github.com/strakam/generals-bots/actions/workflows/codeql.yml)
[![CI](https://github.com/strakam/generals-bots/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/strakam/generals-bots/actions/workflows/pre-commit.yml)

[Installation](#-installation) • [Getting Started](#-getting-started) • [Environment](#-environment) • [Deployment](#-deployment)
</div>

A high-performance JAX-based simulator for [generals.io](https://generals.io), designed for reinforcement learning research.

**Highlights:**
* ⚡ **1M+ steps/second** — fully JIT-compiled JAX simulator with vectorized `vmap` for massive parallelism
* 🎯 **Pure functional design** — immutable state, reproducible trajectories
* 🚀 **Live deployment** — deploy agents to [generals.io](https://generals.io) servers
* 🎮 **Built-in GUI** — visualize games and debug agent behavior

> [!Note]
> This repository is based on the [generals.io](https://generals.io) game.
> The goal is to provide a fast bot development platform for reinforcement learning research.

## 📦 Installation

```bash
git clone https://github.com/strakam/generals-bots
cd generals-bots
pip install -e .
```

## 🌱 Getting Started

### Basic Game Loop

```python
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent

# Create environment (customize grid size and truncation)
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# Create agents
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

# Initialize
key = jrandom.PRNGKey(42)
state = env.reset(key)

# Game loop
while True:
    # Get observations
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    # Get actions
    key, k1, k2 = jrandom.split(key, 3)
    action_0 = agent_0.act(obs_0, k1)
    action_1 = agent_1.act(obs_1, k2)
    actions = jnp.stack([action_0, action_1])

    # Step environment
    key, step_key = jrandom.split(key)
    timestep, state = env.step(state, actions, step_key)

    if timestep.terminated or timestep.truncated:
        break

print(f"Winner: Player {int(timestep.info.winner)}")
```

### ⚡Vectorized Parallel Environments

Run **thousands** of games in parallel using `jax.vmap`:

```python
import jax
import jax.random as jrandom
from generals import GeneralsEnv, get_observation

# Create single environment
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# Vectorize reset and step
NUM_ENVS = 1024
reset_vmap = jax.vmap(env.reset)
step_vmap = jax.vmap(env.step)

# Initialize all environments
key = jrandom.PRNGKey(0)
keys = jrandom.split(key, NUM_ENVS)
states = reset_vmap(keys)  # Batched states

# Step all environments in parallel
# ... get batched observations and actions ...
timesteps, states = step_vmap(states, actions, reset_keys)
```

See `examples/vectorized_example.py` for a complete example.

## 🌍 Environment

### Observation

Each player receives an `Observation` with these fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `armies` | `(H, W)` | Army counts in visible cells |
| `generals` | `(H, W)` | Mask of visible generals |
| `cities` | `(H, W)` | Mask of visible cities |
| `mountains` | `(H, W)` | Mask of visible mountains |
| `owned_cells` | `(H, W)` | Mask of cells you own |
| `opponent_cells` | `(H, W)` | Mask of opponent's visible cells |
| `neutral_cells` | `(H, W)` | Mask of neutral visible cells |
| `fog_cells` | `(H, W)` | Mask of fog (unexplored) cells |
| `structures_in_fog` | `(H, W)` | Mask of cities/mountains in fog |
| `owned_land_count` | scalar | Total cells you own |
| `owned_army_count` | scalar | Total armies you have |
| `opponent_land_count` | scalar | Opponent's cell count |
| `opponent_army_count` | scalar | Opponent's army count |
| `timestep` | scalar | Current game step |

### Action

Actions are arrays of 5 integers: `[pass, row, col, direction, split]`

| Index | Field | Values |
|-------|-------|--------|
| 0 | `pass` | `1` to pass, `0` to move |
| 1 | `row` | Source cell row |
| 2 | `col` | Source cell column |
| 3 | `direction` | `0`=up, `1`=down, `2`=left, `3`=right |
| 4 | `split` | `1` to send half army, `0` to send all-1 |

Use `compute_valid_move_mask` to get legal moves:

```python
from generals import compute_valid_move_mask

mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
# mask shape: (H, W, 4) - True where move from (i,j) in direction d is valid
```

## 🚀 Deployment

Deploy agents to live [generals.io](https://generals.io) servers:

```python
from generals.remote import autopilot
from generals.agents import ExpanderAgent

agent = ExpanderAgent()
autopilot(agent, user_id="your_user_id", lobby_id="your_lobby")
```

Register at [generals.io](https://generals.io) to get your user ID.

## 📄 Citation

```bibtex
@misc{generals_rl,
      author    = {Matej Straka, Martin Schmid},
      title     = {Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning},
      year      = {2025},
      eprint    = {2507.06825},
      archivePrefix = {arXiv},
      primaryClass = {cs.LG},
}
```
