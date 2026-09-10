"""Single-episode helpers shared by local competition and hosted adapters.

Unlike the RL interface, these never allocate an auto-reset state pool.
"""

import jax.numpy as jnp
import jax.random as jrandom

from generals.core import game
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid
from generals.modifiers import build_castles as _bc
from generals.modifiers import deathtouch as _dt


def make_board(env, seed):
    """Build ONE starting board for `env` from `seed`.

    env.reset() would generate a 10k-state training pool, and env.init_state()
    always hands back the largest square board. Under a variable-size ruleset,
    draw each side independently, as the competition evaluator does, rather
    than making every episode use the maximum-size square.
    """
    key = jrandom.PRNGKey(seed)
    if env._fixed_dims is not None:
        return env.init_state(key)

    kd, kg = jrandom.split(key)
    lo, hi = env.min_grid_size, env.max_grid_size
    h = int(jrandom.randint(kd, (), lo, hi + 1))
    w = int(jrandom.randint(jrandom.fold_in(kd, 1), (), lo, hi + 1))
    grid = generate_grid(
        kg,
        grid_dims=(h, w),
        mountain_density_range=env.mountain_density_range,
        num_castles_range=env.num_castles_range,
        # the ruleset's spawn floor, in walking steps around the mountains;
        # generate_grid pads bottom/right for pooling, so the slice below trims
        # it back to the exact rectangle.
        min_generals_distance=env.min_generals_distance,
        castle_val_range=env.castle_val_range,
    )[:h, :w]
    if env.build_castles:
        # nothing neutral to capture — every castle in the game gets built
        grid = _bc.strip_neutral_castles(grid)
    return create_initial_state(grid.astype(jnp.int32))


def make_transition(env):
    """Return the env's per-turn transition: ruleset modifiers + the base step.

    `game.step` is only the base game. The modifiers a ruleset switches on are
    applied *around* it, and skipping them silently changes the rules rather
    than erroring: `game.step` tests `pass == 1`, so a build action
    ([2, row, col, ...]) falls through to a plain move and marches the army out
    of the cell instead of building, and no deathtouch ever fires. This is the
    same composition env.step() performs, minus its vectorised-training pool.
    """

    def transition(state, actions):
        if env.build_castles:
            # builds resolve first and come back rewritten as passes
            state, actions = _bc.apply_build_actions(state, actions)
        if env.deathtouch_turn is not None:
            return _dt.step(state, actions, env.deathtouch_turn)
        return game.step(state, actions)

    return transition
