#!/bin/bash
# The job behind the GPU row of the paper's throughput table and Table S.VI: the 1v1 sweep, then
# 2v2 and four-player free-for-all at 65,536 environments, then a 524,288-environment memory probe.
#SBATCH --job-name=genbench_gpu
#SBATCH --partition=h200fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x_%j.out
set -x
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ID=${SLURM_JOB_ID}
echo "== 1v1 sweep"
.venv-jax/bin/python scripts/bench_jax.py --seconds 15 --envs 1 16 64 256 1024 4096 16384 65536 131072 262144 --tag v3-h200 | tee results/v3_jax_gpu_${ID}.csv
echo "== 2v2";                   .venv-jax/bin/python scripts/bench_jax.py --seconds 12 --envs 65536 --tag v3-h200-2v2  --teams 0,0,1,1 | tee results/v3_jax_gpu_2v2_${ID}.csv
echo "== 4-player free-for-all"; .venv-jax/bin/python scripts/bench_jax.py --seconds 12 --envs 65536 --tag v3-h200-ffa4 --players 4 | tee results/v3_jax_gpu_ffa4_${ID}.csv
echo "== 524288 (memory probe)"; .venv-jax/bin/python scripts/bench_jax.py --seconds 15 --envs 524288 --tag v3-h200-probe | tee results/v3_jax_gpu_probe_${ID}.csv
nvidia-smi --query-gpu=memory.used --format=csv
echo done
