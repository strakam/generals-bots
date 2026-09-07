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

| Parallel envs | 1 | 16 | 64 | 256 | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---|---|---|---|---|---|---|---|
| NumPy, CPU node | 1.1k | 6.1k | 8.7k | 10.6k | | | | |
| JAX, same CPU node | 9.6k | 65k | 92k | 183k | 364k | 476k | 497k | 734k |
| JAX, one H200 | 21k | 183k | 875k | 2.8M | 10.3M | 20.5M | 35.6M | 45.0M |

Four-player modes on one H200 (env_step, 65,536 envs): 2v2 19.4M, free-for-all 19.4M
(about 43% of the 1v1 figure of 44.9M measured in the same job).

## Reproduce

```bash
# JAX simulator (CPU or GPU is picked by JAX; JAX_PLATFORMS=cpu forces CPU)
python paper/benchmarks/bench_jax.py --seconds 15 --envs 1 16 64 256 1024 4096 16384 65536 --tag mytag
python paper/benchmarks/bench_jax.py --teams 0,0,1,1 --envs 1024 65536 --tag 2v2      # 2v2
python paper/benchmarks/bench_jax.py --players 4  --envs 1024 65536 --tag ffa4         # free-for-all

# NumPy simulator: install generals-bots at commit b1636da in a separate venv, then
python paper/benchmarks/bench_numpy.py --seconds 15 --procs 1 16 32 64 128 256 --tag mytag

# SLURM job files used on the RCI cluster (paths relative to paper/benchmarks/)
sbatch paper/benchmarks/slurm/cpu_wholenode.slurm   # NumPy + JAX-CPU, 3 repetitions
sbatch paper/benchmarks/slurm/gpu_sweep.slurm       # JAX on one H200
sbatch paper/benchmarks/slurm/gpu_modes.slurm       # 1v1 / 2v2 / FFA on one H200

# Tables and figure from the CSVs
python paper/benchmarks/make_tables.py paper/benchmarks/results/*.csv
python paper/benchmarks/plot_throughput.py --numpy-tag amd-excl-r1,amd-excl-r2,amd-excl-r3 \
    --jaxcpu-tag amd-excl-r1,amd-excl-r2,amd-excl-r3 --gpu-tag h200-merged --out sim_throughput.pdf
```

`results/` holds the CSVs used in the paper; `results/other-runs/` holds earlier runs on
shared nodes and pre-release code (run-to-run variation on shared CPU nodes was 10-80%,
which is why the paper uses whole-node medians).
