# Experimental Examples

This folder contains experimental code and advanced examples that are not yet part of the main API.

## Contents

### `benchmark_performance.py`
Performance benchmark for measuring throughput with vectorized environments.

```bash
python benchmark_performance.py [num_envs] [num_steps] [iterations]

# Examples
python benchmark_performance.py 256 100 5      # 256 envs, 100 steps, 5 iterations
python benchmark_performance.py 1024 500 3     # 1024 envs, 500 steps, 3 iterations
python benchmark_performance.py 4096 100 2     # 4096 envs, 100 steps, 2 iterations
```

Outputs throughput statistics (steps/s) and helps identify performance bottlenecks.

### `ppo/`
Experimental PPO (Proximal Policy Optimization) training implementation. Work in progress.

### `visualize_policy.py`
Visualization tools for trained policies. Work in progress.
