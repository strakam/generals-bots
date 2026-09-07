#!/bin/bash
#SBATCH --job-name=genbench_cpu_excl
#SBATCH --partition=amdfast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
set -x
cd "$(dirname "$0")/.."
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"; echo "allocated threads: $(nproc) (SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK) whole node via full memory request"
ID=${SLURM_JOB_ID}
for R in 1 2 3; do
  TAG=amd-excl-r${R}
  echo "== rep $R: NumPy"; .venv-numpy/bin/python scripts/bench_numpy.py --seconds 12 --procs 1 16 32 64 128 256 --tag $TAG | tee results/numpy_cpu_${ID}_r${R}.csv
  echo "== rep $R: JAX CPU"; JAX_PLATFORMS=cpu .venv-jax/bin/python scripts/bench_jax.py --seconds 12 --envs 1 16 64 256 1024 4096 16384 65536 --tag $TAG | tee results/jax_cpu_${ID}_r${R}.csv
done
echo done
