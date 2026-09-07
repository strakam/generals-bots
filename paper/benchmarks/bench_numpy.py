"""Throughput sweep of the prior NumPy/numba Generals.io simulator (generals-bots, PyTorch era).

Frame = one environment advanced by one game turn (both players act once).
Parallelism = one gym AsyncVectorEnv with P worker processes (the setup of the
prior paper: 12 processes on a 12-core CPU). Actions are uniform samples from
the gym action space, as in the repo's own tests/test_performance.py.
Prints one CSV line per process count.
"""
import argparse, os, socket, sys, time
import numpy as np
import gymnasium as gym
from generals import GridFactory, GymnasiumGenerals


def make_env(h, w, truncation):
    gf = GridFactory(min_grid_dims=(h, w), max_grid_dims=(h, w))
    return lambda: GymnasiumGenerals(agents=["a0", "a1"], grid_factory=gf, truncation=truncation)


def run(procs, h, w, seconds, truncation=500):
    fns = [make_env(h, w, truncation) for _ in range(procs)]
    envs = gym.vector.SyncVectorEnv(fns) if procs == 1 else gym.vector.AsyncVectorEnv(fns, context="forkserver")
    envs.reset(seed=0)
    space = envs.single_action_space
    def act():
        return np.stack([[space.sample() for _ in range(procs)], [space.sample() for _ in range(procs)]], axis=1)
    for _ in range(20):
        envs.step(act())
    frames, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        envs.step(act()); frames += procs
    dt = time.perf_counter() - t0
    envs.close()
    return frames / dt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--procs", type=int, nargs="+", default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    procs = args.procs or [q for q in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128] if q <= cores]
    print(f"# host={socket.gethostname()} cores_available={cores} python={sys.version.split()[0]} grid={args.grid} tag={args.tag}", flush=True)
    print("sim,device,envs,variant,frames_per_s", flush=True)
    for q in procs:
        fps = run(q, args.grid, args.grid, args.seconds)
        print(f"numpy,cpu,{q},env_step,{fps:.0f}", flush=True)
