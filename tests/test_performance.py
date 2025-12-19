"""Performance test for parallel environments."""
import time

import gymnasium as gym
import numpy as np

from generals import GridFactory, GymnasiumGenerals


def test_performance_parallel_envs(num_envs=16, target_total_steps=1_000_000):
    """
    Test performance of parallel environments with random actions.
    Calculates time to complete 1 million total steps across all environments.
    
    Args:
        num_envs: Number of parallel environments to run (default: 16)
        target_total_steps: Total number of steps to complete (default: 1,000,000)
    """
    agent_names = ["agent_0", "agent_1"]
    
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
    )
    
    # Create parallel environments
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: GymnasiumGenerals(
                agents=agent_names, 
                grid_factory=grid_factory, 
                truncation=500
            )
            for _ in range(num_envs)
        ],
    )
    
    envs.reset()
    
    total_steps = 0
    start_time = time.time()
    
    while total_steps < target_total_steps:
        # Sample random actions for each agent in each environment
        agent_actions = [envs.single_action_space.sample() for _ in range(num_envs)]
        npc_actions = [envs.single_action_space.sample() for _ in range(num_envs)]
        
        # Stack actions together -> shape: (num_envs, n_agents, action_dim)
        actions = np.stack([agent_actions, npc_actions], axis=1)
        
        observations, _, terminated, truncated, infos = envs.step(actions)
        
        total_steps += num_envs
        
        # Reset any environments that finished
        if any(terminated) or any(truncated):
            # AsyncVectorEnv automatically resets terminated/truncated environments
            pass
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    envs.close()
    
    # Calculate and print performance metrics
    steps_per_second = target_total_steps / elapsed_time
    
    print(f"\nPerformance Test Results:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Total steps completed: {total_steps:,}")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Steps per second: {steps_per_second:.2f}")
    print(f"  Time per step: {elapsed_time / total_steps * 1000:.4f} ms")
    
    # Assert test completed successfully
    assert total_steps >= target_total_steps
    assert elapsed_time > 0
    
    return {
        "num_envs": num_envs,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "steps_per_second": steps_per_second,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance test for parallel environments")
    parser.add_argument(
        "--num-envs", 
        type=int, 
        default=16, 
        help="Number of parallel environments (default: 16)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1_000_000, 
        help="Total number of steps to complete (default: 1,000,000)"
    )
    
    args = parser.parse_args()
    
    test_performance_parallel_envs(
        num_envs=args.num_envs, 
        target_total_steps=args.steps
    )
