"""Batched multiplayer games: 2v2, 4-player free-for-all and the classic 1v1.

For each mode this builds one env, resets it, and steps a vmapped batch of
environments for a fixed number of turns under jax.jit. Every player plays a
random VALID action each turn (generals.core.action.sample_valid_action, run
once per player on that player's own fog-of-war observation), finished games
auto-reset from the pool, and at the end it prints how many games finished and
which team won them.

    python examples/multiplayer_example.py                  # 64 envs x 1000 turns
    python examples/multiplayer_example.py --envs 256 --turns 2000
"""
import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.core.action import sample_valid_action

MODES = {
    # name: (constructor kwargs, short description)
    "1v1": (dict(), "2 players, the default"),
    "2v2": (dict(teams=[0, 0, 1, 1]), "players 0+1 vs 2+3"),
    "ffa4": (dict(num_players=4), "4-player free-for-all"),
}


def run(name, env_kwargs, num_envs, num_turns, grid, truncation, seed):
    env = GeneralsEnv(grid_dims=(grid, grid), truncation=truncation, pool_size=max(2 * num_envs, 64), **env_kwargs)
    n = env.num_players
    teams = [int(t) for t in env.teams]

    key = jrandom.PRNGKey(seed)
    key, k_pool = jrandom.split(key)
    pool, _ = env.reset(k_pool)
    # Each env starts on its own pool board and walks the pool from there.
    states = jax.tree.map(lambda x: x[:num_envs], pool)
    states = states._replace(pool_idx=jnp.arange(num_envs, dtype=jnp.int32) + num_envs)

    def act(state, keys):
        """One env: a random valid action for each of the n players."""
        return jnp.stack([sample_valid_action(keys[i], get_observation(state, i)) for i in range(n)])

    vstep = jax.vmap(env.step, in_axes=(0, 0, None))

    @jax.jit
    def play(states, key):
        def body(carry, _):
            states, key = carry
            key, k = jrandom.split(key)
            keys = jrandom.split(k, num_envs * n).reshape(num_envs, n, -1)
            actions = jax.vmap(act)(states, keys)                  # (num_envs, n, 5)
            before = states
            ts, states = vstep(states, actions, pool)
            done = ts.terminated | ts.truncated
            # generals captured this turn (ts.last_state is the board before any auto-reset)
            captured = (ts.last_state.eliminated & ~before.eliminated).sum()
            # winner is a team id (-1 while running / on a draw); count per team
            won = (ts.info.winner[:, None] == jnp.arange(n)[None, :]) & ts.terminated[:, None]
            stats = jnp.concatenate([
                jnp.array([done.sum(), ts.terminated.sum(), (ts.truncated & ~ts.terminated).sum(), captured]),
                won.sum(axis=0),
            ])
            return (states, key), stats
        (states, key), stats = jax.lax.scan(body, (states, key), None, length=num_turns)
        return states, stats.sum(axis=0)

    t0 = time.perf_counter()
    states, stats = play(states, key)
    jax.block_until_ready(stats)
    compile_and_run = time.perf_counter() - t0
    t0 = time.perf_counter()
    states, stats = play(states, key)
    jax.block_until_ready(stats)
    elapsed = time.perf_counter() - t0

    finished, by_capture, by_truncation, captured = (int(x) for x in stats[:4])
    wins = [int(x) for x in stats[4:]]
    team_ids = sorted(set(teams))
    frames = num_envs * num_turns
    print(f"[{name}] {MODES[name][1]}: teams={teams}, grid {grid}x{grid}, "
          f"{num_envs} envs x {num_turns} turns under jit "
          f"({frames / elapsed:,.0f} env-steps/s; first call incl. compile {compile_and_run:.1f}s)")
    print(f"[{name}]   generals captured: {captured};  games finished: {finished}  "
          f"(last team standing: {by_capture}, truncated at {truncation} turns: {by_truncation})")
    for t in team_ids:
        members = [i for i, tt in enumerate(teams) if tt == t]
        print(f"[{name}]   team {t} (players {members}) won {wins[t]}")
    alive = int((~states.eliminated).sum())
    print(f"[{name}]   still running: {num_envs} games, {alive} of {num_envs * n} players alive")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--turns", type=int, default=1000)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--truncation", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for name in args.modes:
        run(name, MODES[name][0], args.envs, args.turns, args.grid, args.truncation, args.seed)
