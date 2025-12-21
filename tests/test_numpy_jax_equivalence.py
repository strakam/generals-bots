"""
Minimal correctness test: Compare NumPy environment vs JAX environment step-by-step.

This test ensures that the JAX implementation produces EXACTLY the same results
as the NumPy implementation for the same grid and action sequence.
"""

import jax
import jax.numpy as jnp
import numpy as np

from generals.core.grid import Grid, GridFactory
from generals.envs.gymnasium_generals import GymnasiumGenerals
from generals.envs.jax_env import VectorizedJaxEnv


def test_numpy_vs_jax_correctness():
    """Test that NumPy and JAX envs produce identical results."""
    
    # Use same grid factory for both
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
        mountain_density=0.1,
        city_density=0.1,
        general_positions=[[1, 1], [8, 8]],
        seed=42,  # Fixed seed for determinism
    )
    
    # Create NumPy environment
    numpy_env = GymnasiumGenerals(
        agents=["player_0", "player_1"],
        grid_factory=grid_factory,
        truncation=500,
    )
    
    # Create JAX environment (single env for comparison)
    jax_env = VectorizedJaxEnv(
        num_envs=1,
        grid_size=(10, 10),
        mountain_density=0.1,
        city_density=0.1,
        render_mode=None,
    )
    
    # Reset both environments
    numpy_obs, numpy_info = numpy_env.reset(seed=42)
    jax_reset_output = jax_env.reset(seed=42)
    print(f"JAX reset output type: {type(jax_reset_output)}")
    print(f"JAX reset output length: {len(jax_reset_output)}")
    jax_obs, jax_info = jax_reset_output
    print(f"JAX obs type: {type(jax_obs)}, shape: {jax_obs.shape if hasattr(jax_obs, 'shape') else 'N/A'}")
    
    print(f"NumPy obs type: {type(numpy_obs)}")
    if hasattr(numpy_obs, 'keys'):
        print(f"NumPy obs keys: {list(numpy_obs.keys())}")
    if hasattr(numpy_obs, 'shape'):
        print(f"NumPy obs shape: {numpy_obs.shape}")
    
    print("=" * 80)
    print("INITIAL STATE COMPARISON")
    print("=" * 80)
    
    # Compare initial observations
    compare_observations(numpy_obs, jax_obs, step=0)
    compare_info(numpy_info, jax_info, step=0)
    
    # Run 20 steps with predefined actions
    num_steps = 20
    np.random.seed(42)
    
    print("\n" + "=" * 80)
    print("RUNNING 20 STEPS")
    print("=" * 80)
    
    for step in range(num_steps):
        # Generate same random action for both
        # Action format: [pass_flag, src_row, src_col, dst_row, dst_col]
        action_p0 = np.random.randint(0, 10, size=5, dtype=np.int32)
        action_p1 = np.random.randint(0, 10, size=5, dtype=np.int32)
        
        # NumPy env expects dict of actions
        numpy_action = {
            "player_0": action_p0,
            "player_1": action_p1,
        }
        
        # JAX env expects (num_envs, 2, 5) array
        jax_action = jnp.array([[action_p0, action_p1]], dtype=jnp.int32)
        
        # Step both environments
        numpy_obs, numpy_reward, numpy_terminated, numpy_truncated, numpy_info = numpy_env.step(numpy_action)
        jax_obs, jax_reward, jax_terminated, jax_truncated, jax_info = jax_env.step(jax_action)
        
        print(f"\n--- Step {step + 1} ---")
        print(f"Actions: P0={action_p0}, P1={action_p1}")
        
        # Compare observations
        compare_observations(numpy_obs, jax_obs, step=step+1)
        
        # Compare rewards
        compare_rewards(numpy_reward, jax_reward, step=step+1)
        
        # Compare termination flags
        compare_flags(numpy_terminated, numpy_truncated, jax_terminated, jax_truncated, step=step+1)
        
        # Compare info
        compare_info(numpy_info, jax_info, step=step+1)
        
        # If terminated, break
        if numpy_terminated["player_0"] or numpy_truncated["player_0"]:
            print(f"\nEnvironment terminated at step {step + 1}")
            break
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - NumPy and JAX implementations are identical!")
    print("=" * 80)


def compare_observations(numpy_obs, jax_obs, step):
    """Compare observation dictionaries."""
    # JAX obs is ObservationJax NamedTuple with fields (armies, generals, cities, etc)
    # Each field has shape (num_envs, 2, H, W) for the 2 players
    # NumPy obs is dict with keys "player_0" and "player_1", each (7, H, W)
    
    # Extract player observations
    numpy_obs_p0 = numpy_obs["player_0"]  # (7, H, W)
    numpy_obs_p1 = numpy_obs["player_1"]  # (7, W)
    
    # JAX observation fields
    jax_armies = np.array(jax_obs.armies[0])  # (2, H, W)
    jax_generals = np.array(jax_obs.generals[0])  # (2, H, W)
    jax_cities = np.array(jax_obs.cities[0])  # (2, H, W)
    jax_mountains = np.array(jax_obs.mountains[0])  # (2, H, W)
    jax_fog = np.array(jax_obs.fog[0])  # (2, H, W)
    jax_owned = np.array(jax_obs.owned_cells[0])  # (2, H, W)
    jax_opponent = np.array(jax_obs.opponent_cells[0])  # (2, H, W)
    
    # Combine into (2, 7, H, W) format
    jax_obs_combined = np.stack([
        np.stack([jax_armies[i], jax_generals[i], jax_cities[i], jax_mountains[i], 
                  jax_fog[i], jax_owned[i], jax_opponent[i]], axis=0)
        for i in range(2)
    ], axis=0)
    
    jax_obs_p0 = jax_obs_combined[0]  # (7, H, W)
    jax_obs_p1 = jax_obs_combined[1]  # (7, H, W)
    
    # Compare each channel
    channels = ["army", "general", "city", "mountain", "fog", "owned_cells", "opponent_cells"]
    for i, channel_name in enumerate(channels):
        p0_match = np.allclose(numpy_obs_p0[i], jax_obs_p0[i])
        p1_match = np.allclose(numpy_obs_p1[i], jax_obs_p1[i])
        
        if not p0_match or not p1_match:
            print(f"  ✗ Step {step} - Channel {i} ({channel_name}) MISMATCH!")
            if not p0_match:
                diff = np.abs(numpy_obs_p0[i] - jax_obs_p0[i])
                print(f"    Player 0 max diff: {diff.max()}")
                print(f"    NumPy sample:\n{numpy_obs_p0[i][:3, :3]}")
                print(f"    JAX sample:\n{jax_obs_p0[i][:3, :3]}")
            if not p1_match:
                diff = np.abs(numpy_obs_p1[i] - jax_obs_p1[i])
                print(f"    Player 1 max diff: {diff.max()}")
            raise AssertionError(f"Observation mismatch at step {step}")


def compare_rewards(numpy_reward, jax_reward, step):
    """Compare rewards."""
    # JAX reward is (1, 2), NumPy is dict
    jax_r_p0 = float(jax_reward[0, 0])
    jax_r_p1 = float(jax_reward[0, 1])
    
    numpy_r_p0 = float(numpy_reward["player_0"])
    numpy_r_p1 = float(numpy_reward["player_1"])
    
    if not (np.isclose(jax_r_p0, numpy_r_p0) and np.isclose(jax_r_p1, numpy_r_p1)):
        print(f"  ✗ Step {step} - Reward MISMATCH!")
        print(f"    NumPy: P0={numpy_r_p0}, P1={numpy_r_p1}")
        print(f"    JAX:   P0={jax_r_p0}, P1={jax_r_p1}")
        raise AssertionError(f"Reward mismatch at step {step}")


def compare_flags(numpy_term, numpy_trunc, jax_term, jax_trunc, step):
    """Compare termination and truncation flags."""
    # JAX flags are (1,) boolean arrays, NumPy are dicts with bools
    jax_term_val = bool(jax_term[0])
    jax_trunc_val = bool(jax_trunc[0])
    
    # NumPy returns dict with same value for both players
    numpy_term_val = numpy_term["player_0"]
    numpy_trunc_val = numpy_trunc["player_0"]
    
    if jax_term_val != numpy_term_val or jax_trunc_val != numpy_trunc_val:
        print(f"  ✗ Step {step} - Flags MISMATCH!")
        print(f"    NumPy: terminated={numpy_term_val}, truncated={numpy_trunc_val}")
        print(f"    JAX:   terminated={jax_term_val}, truncated={jax_trunc_val}")
        raise AssertionError(f"Termination flags mismatch at step {step}")


def compare_info(numpy_info, jax_info, step):
    """Compare info dictionaries."""
    # JAX info is GameInfo NamedTuple with shape (1,) for each field
    # NumPy info is dict with player keys
    
    # Extract JAX values (first environment)
    jax_valid_p0 = np.array(jax_info.valid_actions_mask[0, 0])  # (max_actions,)
    jax_valid_p1 = np.array(jax_info.valid_actions_mask[0, 1])  # (max_actions,)
    jax_army_count_p0 = int(jax_info.army_count[0, 0])
    jax_army_count_p1 = int(jax_info.army_count[0, 1])
    jax_land_count_p0 = int(jax_info.land_count[0, 0])
    jax_land_count_p1 = int(jax_info.land_count[0, 1])
    
    # Extract NumPy values
    numpy_valid_p0 = numpy_info["player_0"]["valid_actions_mask"]
    numpy_valid_p1 = numpy_info["player_1"]["valid_actions_mask"]
    numpy_army_count_p0 = numpy_info["player_0"]["army_count"]
    numpy_army_count_p1 = numpy_info["player_1"]["army_count"]
    numpy_land_count_p0 = numpy_info["player_0"]["land_count"]
    numpy_land_count_p1 = numpy_info["player_1"]["land_count"]
    
    # Compare valid actions masks (truncate to same length)
    min_len = min(len(jax_valid_p0), len(numpy_valid_p0))
    if not (np.allclose(jax_valid_p0[:min_len], numpy_valid_p0[:min_len]) and 
            np.allclose(jax_valid_p1[:min_len], numpy_valid_p1[:min_len])):
        print(f"  ✗ Step {step} - Valid actions mask MISMATCH!")
        print(f"    NumPy P0 true count: {numpy_valid_p0.sum()}")
        print(f"    JAX P0 true count: {jax_valid_p0.sum()}")
        raise AssertionError(f"Valid actions mask mismatch at step {step}")
    
    # Compare army counts
    if jax_army_count_p0 != numpy_army_count_p0 or jax_army_count_p1 != numpy_army_count_p1:
        print(f"  ✗ Step {step} - Army count MISMATCH!")
        print(f"    NumPy: P0={numpy_army_count_p0}, P1={numpy_army_count_p1}")
        print(f"    JAX:   P0={jax_army_count_p0}, P1={jax_army_count_p1}")
        raise AssertionError(f"Army count mismatch at step {step}")
    
    # Compare land counts
    if jax_land_count_p0 != numpy_land_count_p0 or jax_land_count_p1 != numpy_land_count_p1:
        print(f"  ✗ Step {step} - Land count MISMATCH!")
        print(f"    NumPy: P0={numpy_land_count_p0}, P1={numpy_land_count_p1}")
        print(f"    JAX:   P0={jax_land_count_p0}, P1={jax_land_count_p1}")
        raise AssertionError(f"Land count mismatch at step {step}")


if __name__ == "__main__":
    test_numpy_vs_jax_correctness()
