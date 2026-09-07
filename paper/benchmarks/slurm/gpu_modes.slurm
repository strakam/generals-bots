#!/bin/bash
#SBATCH --job-name=genbench_modes
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
ENVS="1024 4096 16384 65536"
echo "== 1v1 (merged master)";   .venv-jax/bin/python scripts/bench_jax.py --seconds 12 --envs $ENVS --tag h200-1v1  | tee results/jax_gpu_1v1_${ID}.csv
echo "== 2v2";                   .venv-jax/bin/python scripts/bench_jax.py --seconds 12 --envs $ENVS --tag h200-2v2  --teams 0,0,1,1 | tee results/jax_gpu_2v2_${ID}.csv
echo "== 4-player free-for-all"; .venv-jax/bin/python scripts/bench_jax.py --seconds 12 --envs $ENVS --tag h200-ffa4 --players 4 | tee results/jax_gpu_ffa4_${ID}.csv
echo done
