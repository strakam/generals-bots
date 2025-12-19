# JAX Game Implementation

This module provides a high-performance JAX-based implementation of the Generals.io game logic.

## Key Features

- **Pure functional design**: No classes, all pure functions
- **JIT compilation**: Full compatibility with JAX's JIT compiler
- **Batched execution**: Native support for parallel environments via `vmap`
- **Fixed 2-player**: Simplified for performance (no dynamic agent names)
- **Array-based state**: Uses arrays instead of dictionaries for efficient GPU/TPU execution

## Performance

Single environment comparison (10x10 grid, 1000 steps):
- **NumPy/Numba**: ~3,300 steps/sec
- **JAX (single)**: ~6,700 steps/sec (**2x faster**)

Batched execution (10x10 grid):
- **256 parallel envs**: ~7,350 steps/sec per environment
- **1M total steps**: ~134 seconds (vs ~314 seconds with AsyncVectorEnv)

## Usage

### Basic Example

```python
import jax
import jax.numpy as jnp
from generals.core import game_jax

# Create initial state from grid
grid = jnp.array([
    [ord('A'), ord('.'), ord('.')],
    [ord('.'), ord('.'), ord('.')],
    [ord('.'), ord('.'), ord('B')],
], dtype=jnp.uint8)

state = game_jax.create_initial_state(grid)

# Execute actions
actions = jnp.array([
    [0, 0, 0, 3, 0],  # Player 0: move right from (0,0)
    [1, 0, 0, 0, 0],  # Player 1: pass
], dtype=jnp.int32)

new_state, info = game_jax.step(state, actions)

# Get observation for player 0
obs = game_jax.get_observation(new_state, player_idx=0)
```

### Batched Execution

```python
import jax
import jax.numpy as jnp
from generals.core import game_jax

# Create single state
state = game_jax.create_initial_state(grid)

# Batch it
num_envs = 256
batched_state = jax.tree.map(
    lambda x: jnp.stack([x] * num_envs),
    state
)

# JIT compile batched step
jitted_step = jax.jit(game_jax.batch_step)

# Execute batched step
actions = jnp.zeros((num_envs, 2, 5), dtype=jnp.int32)  # All pass
new_states, infos = jitted_step(batched_state, actions)
```

## State Representation

The game state is a dictionary with the following arrays:

- `armies`: [H, W] - army count per cell
- `ownership`: [2, H, W] - player ownership masks
- `ownership_neutral`: [H, W] - neutral cells mask
- `generals`: [H, W] - general positions (static)
- `cities`: [H, W] - city positions (static)
- `mountains`: [H, W] - mountain positions (static)
- `passable`: [H, W] - passable cells (static)
- `general_positions`: [2, 2] - (row, col) for each general
- `time`: scalar - current timestep
- `winner`: scalar - winner index (-1 = none, 0/1 = player)

## Action Format

Actions are `[5]` integer arrays: `[pass, row, col, direction, split]`

- `pass`: 1 to skip turn, 0 to move
- `row`, `col`: source cell coordinates
- `direction`: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
- `split`: 1 to split army, 0 to move all-but-one

## Differences from NumPy Implementation

1. **Fixed 2 players**: No dynamic agent names or variable player counts
2. **Array-based**: Ownership is `[2, H, W]` not `dict[str, array]`
3. **Simplified ordering**: Basic player priority (can be extended)
4. **No Channels class**: State is flat dictionary of arrays
5. **Functional**: Pure functions instead of class methods

## Limitations

- Currently 2 players only (could extend to N players)
- Simplified agent ordering logic
- No dynamic grid generation (reuse NumPy GridFactory, then convert)
- Basic visibility calculation (could optimize further)

## Future Improvements

- [ ] Support for N players
- [ ] More sophisticated agent ordering
- [ ] GPU/TPU optimization
- [ ] Integration with Gymnasium/PettingZoo APIs
- [ ] Proper action masking computation
