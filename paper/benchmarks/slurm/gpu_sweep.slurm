#!/bin/bash
#SBATCH --job-name=genbench_gpu_merged
#SBATCH --partition=h200fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=logs/%x_%j.out
set -x
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ID=${SLURM_JOB_ID}
echo "== JAX simulator on H200: shared list"
.venv-jax/bin/python scripts/bench_jax.py --seconds 15 --envs 1 16 64 256 1024 4096 16384 65536 --tag h200-merged | tee results/jax_gpu_${ID}.csv
echo done
