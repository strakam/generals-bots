"""Throughput sweep of the JAX Generals.io simulator (works on CPU or GPU).

Frame = one environment advanced by one game turn (both players act once).
Variants (all include auto-reset from the state pool):
  step_only : raw game step, no observations, cheap uniform-random actions
  env_step  : env.step = game step + fog-of-war observations for BOTH players,
              cheap uniform-random actions (same semantics as the old env's
              gym action_space.sample(): random cell/direction/split, 10% pass)
  env_valid : env.step + the repo's random-VALID-action sampler (mask + argwhere)
              -- optional, dominated by the sampler, off by default
Compile time is excluded (one warm-up call per configuration).
Prints one CSV line per (envs, variant).
"""
import argparse, functools, os, platform, socket, sys, time
import jax, jax.numpy as jnp, jax.random as jrandom
from generals import GeneralsEnv
from generals.core.game import step as game_step
from generals.core.action import sample_valid_action


def uniform_actions(key, n, h, w, players=2):
    k1, k2, k3, k4, k5 = jrandom.split(key, 5)
    to_pass = (jrandom.uniform(k1, (n, players)) < 0.1).astype(jnp.int32)
    row = jrandom.randint(k2, (n, players), 0, h)
    col = jrandom.randint(k3, (n, players), 0, w)
    direction = jrandom.randint(k4, (n, players), 0, 4)
    split = jrandom.randint(k5, (n, players), 0, 2)
    return jnp.stack([to_pass, row, col, direction, split], axis=-1).astype(jnp.int32)


def make(n, grid, pool_size, players=2, teams=None):
    kw = {}
    if teams is not None: kw["teams"] = teams
    elif players != 2: kw["num_players"] = players
    env = GeneralsEnv(grid_dims=(grid, grid), truncation=500, pool_size=pool_size, **kw)
    P = len(teams) if teams is not None else players
    pool, _ = env.reset(jrandom.PRNGKey(0))
    states = jax.tree.map(lambda x: x[:n], pool)
    states = states._replace(pool_idx=jnp.arange(n, dtype=states.pool_idx.dtype) + n)
    vstep = jax.vmap(env.step, in_axes=(0, 0, None))
    vgame = jax.vmap(game_step)

    @functools.partial(jax.jit, static_argnums=1)
    def run_step_only(carry, steps):
        def body(c, _):
            s, key = c
            key, k = jrandom.split(key)
            s, info = vgame(s, uniform_actions(k, n, grid, grid, P))
            return (s, key), info.winner.sum()
        c, r = jax.lax.scan(body, carry, None, length=steps)
        return c, r.sum()

    @functools.partial(jax.jit, static_argnums=1)
    def run_env_step(carry, steps):
        def body(c, _):
            s, key = c
            key, k = jrandom.split(key)
            ts, s = vstep(s, uniform_actions(k, n, grid, grid, P), pool)
            return (s, key), ts.reward.sum()
        c, r = jax.lax.scan(body, carry, None, length=steps)
        return c, r.sum()

    @functools.partial(jax.jit, static_argnums=1)
    def run_env_valid(carry, steps):
        def body(c, _):
            s, key, acts = c
            ts, s = vstep(s, acts, pool)
            key, k = jrandom.split(key)
            keys = jrandom.split(k, n * P).reshape(n, P, 2)
            acts = jax.vmap(jax.vmap(sample_valid_action))(keys, ts.observation)
            return (s, key, acts), ts.reward.sum()
        c, r = jax.lax.scan(body, carry, None, length=steps)
        return c, r.sum()

    return states, run_step_only, run_env_step, run_env_valid


def timeit(fn, arg, steps, n, target_s):
    out = fn(arg, steps); jax.block_until_ready(out)      # warm-up / compile
    frames, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < target_s:
        out = fn(arg, steps); jax.block_until_ready(out); frames += n * steps
    return frames / (time.perf_counter() - t0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--envs", type=int, nargs="+", default=[1, 16, 128, 1024, 4096, 16384])
    p.add_argument("--valid", action="store_true", help="also run the env_valid variant")
    p.add_argument("--tag", default="")
    p.add_argument("--players", type=int, default=2, help="N-player free-for-all (default 2 = classic 1v1)")
    p.add_argument("--teams", default=None, help="comma-separated team ids per player, e.g. 0,0,1,1 for 2v2")
    args = p.parse_args()
    teams = [int(x) for x in args.teams.split(",")] if args.teams else None
    players = len(teams) if teams else args.players
    dev = jax.devices()[0]
    hdr = dict(host=socket.gethostname(), device=f"{dev.platform}:{getattr(dev, 'device_kind', '')}",
               cores=os.cpu_count(), jax=jax.__version__, python=sys.version.split()[0], grid=args.grid, tag=args.tag,
               players=players, teams=("-".join(map(str, teams)) if teams else "ffa"))
    print("# " + " ".join(f"{k}={v}" for k, v in hdr.items()), flush=True)
    print("sim,device,envs,variant,frames_per_s", flush=True)
    for n in args.envs:
        steps = max(10, min(200, 40000 // n))
        states, f_step, f_env, f_valid = make(n, args.grid, pool_size=max(2 * n, 64), players=players, teams=teams)
        key = jrandom.PRNGKey(1)
        r1 = timeit(f_step, (states, key), steps, n, args.seconds)
        print(f"jax,{dev.platform},{n},step_only,{r1:.0f}", flush=True)
        r2 = timeit(f_env, (states, key), steps, n, args.seconds)
        print(f"jax,{dev.platform},{n},env_step,{r2:.0f}", flush=True)
        if args.valid:
            acts0 = uniform_actions(key, n, args.grid, args.grid, players)
            r3 = timeit(f_valid, (states, key, acts0), steps, n, args.seconds)
            print(f"jax,{dev.platform},{n},env_valid,{r3:.0f}", flush=True)
