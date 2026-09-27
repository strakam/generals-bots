# Simulator throughput benchmarks

Scripts, job files, and raw results behind the throughput figure and table in the paper
*Superhuman AI for Generals.io Using Self-Play Reinforcement Learning* (IEEE ToG revision).

## Definitions

* **Frame**: one environment advanced by one game turn, with every player acting once.
  A batched call that advances *N* environments by one turn counts as *N* frames.
  For the NumPy simulator one process runs exactly one environment, so "parallel
  environments" is the same axis for both simulators.
* **env_step**: `GeneralsEnv.step` = game step + auto-reset from the state pool +
  fog-of-war observations for every player. This is the number reported in the paper.
  The benchmark loop keeps every step's observations in its `lax.scan` carry and returns a
  checksum of them, so XLA cannot drop their computation, and the game states are carried
  across timed calls (the games keep running, as in the NumPy benchmark). An earlier version
  of the loop returned only the reward sum; XLA then removed the observations and the loop
  timed the game transition alone.
* **step_only**: the raw game step without observations (reported in the supplement).
* Actions are uniform random samples (random cell, direction, split; 10% pass) for both
  simulators, so the work per frame is the same. Compile time is excluded.
* Maps are 24x24, truncation 500 turns.

## What was compared

| Simulator | Code | Hardware |
|---|---|---|
| NumPy simulator (prior work, PyTorch era) | `generals-bots` at commit `b1636da` (April 2025, the code the prior paper benchmarked) | one AMD EPYC 7543 node, 64 threads, whole node |
| JAX simulator (this repo) | this repository at the tagged release | the same node; and one NVIDIA H200 NVL (driver 575.51, jax 0.11.1) |

## Results (medians of 3 repetitions on the CPU node; single run on the GPU)

Frames per second, env_step:

| Parallel envs | 1 | 16 | 32 | 64 | 256 | 1,024 | 4,096 | 16,384 | 65,536 | 131,072 | 262,144 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NumPy, CPU node | 1.1k | 5.8k | 7.7k | 8.8k | | | | | | | |
| JAX, same CPU node | 8.5k | 48k | | 84k | 155k | 411k | 474k | 548k | 684k | | |
| JAX, one H200 | 16k | 198k | | 757k | 2.5M | 8.8M | 17.9M | 31.4M | 38.1M | 40.8M | 42.8M |

The NumPy simulator runs one process per allocated thread (64). Oversubscribing it gave no
reliable gain: with 256 processes the three repetitions ranged from 5.7k to 10.4k.
Four-player modes on one H200 (env_step, 65,536 envs, same job as the 1v1 sweep): 2v2 18.6M,
free-for-all 18.6M (1v1 38.1M). 524,288 envs exceed the GPU memory.

## Reproduce

```bash
# JAX simulator (CPU or GPU is picked by JAX; JAX_PLATFORMS=cpu forces CPU)
python paper/benchmarks/bench_jax.py --seconds 15 --envs 1 16 64 256 1024 4096 16384 65536 --tag mytag
python paper/benchmarks/bench_jax.py --teams 0,0,1,1 --envs 65536 --tag 2v2      # 2v2
python paper/benchmarks/bench_jax.py --players 4  --envs 65536 --tag ffa4         # free-for-all

# NumPy simulator: install generals-bots at commit b1636da in a separate venv, then
python paper/benchmarks/bench_numpy.py --seconds 15 --procs 1 16 32 64 128 256 --tag mytag

# SLURM job files used on the RCI cluster (paths relative to paper/benchmarks/)
sbatch paper/benchmarks/slurm/cpu_wholenode.slurm   # NumPy + JAX-CPU, 3 repetitions
sbatch paper/benchmarks/slurm/gpu_sweep.slurm       # JAX on one H200: 1v1 sweep, 2v2, FFA

# Tables and figure from the CSVs
python paper/benchmarks/make_tables.py paper/benchmarks/results/*.csv
python paper/benchmarks/plot_throughput.py --numpy-tag v2-amd-excl-r1,v2-amd-excl-r2,v2-amd-excl-r3 \
    --jaxcpu-tag v2-amd-excl-r1,v2-amd-excl-r2,v2-amd-excl-r3 --gpu-tag v3-h200 --numpy-max-procs 64 \
    --out sim_throughput.pdf
```

`results/` holds the CSVs used in the paper; `results/other-runs/` holds earlier runs (the old
benchmark loop, shared nodes, and pre-release code) (run-to-run variation on shared CPU nodes was 10-80%,
which is why the paper uses whole-node medians).
