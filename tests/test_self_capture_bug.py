"""
CRITICAL BUG TEST: Self-capture bug

Tests that moving onto your own general does NOT end the game.
This was a critical bug where general_captured didn't check ~moving_to_own.
"""
import jax.numpy as jnp
import numpy as np
from generals.core.grid import GridFactory
from generals.core.game_jax import create_initial_state, step as jax_step

print("\n" + "="*80)
print("CRITICAL: SELF-CAPTURE BUG TEST")
print("="*80)
print("\nThis test verifies the fix for the bug where a player could")
print("'capture' their own general by moving armies onto it.")
print("="*80 + "\n")

# Create minimal grid - just 2 adjacent cells with one general
grid_factory = GridFactory(
    min_grid_dims=(3, 3),
    max_grid_dims=(3, 3),
    mountain_density=0.0,
    city_density=0.0,
    general_positions=[[1, 1], [2, 2]],
    seed=999,
)

grid = grid_factory.generate()
print("Grid layout:")
for row in grid.grid:
    print("  ", " ".join(row))
print()

grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
state = create_initial_state(jnp.array(grid_array))

# Verify initial state
assert int(np.array(state.winner)) == -1, "Should start with no winner"
print("✓ Initial state: No winner\n")

# Pass 10 turns to build up armies on generals
for i in range(10):
    state, _ = jax_step(state, jnp.array([[1,0,0,0,0], [1,0,0,0,0]], dtype=jnp.int32))

p0_gen_armies = int(np.array(state.armies[1, 1]))
print(f"After 10 turns: P0 general has {p0_gen_armies} armies\n")

# P0 moves from general (1,1) right to (1,2)
print("Step 1: P0 moves FROM general (1,1) to right (1,2)")
state, _ = jax_step(state, jnp.array([[0,1,1,2,0], [1,0,0,0,0]], dtype=jnp.int32))

# Check if P0 owns (1,2) now
p0_owns_12 = bool(np.array(state.ownership[0, 1, 2]))
armies_12 = int(np.array(state.armies[1, 2]))
print(f"  Result: P0 owns (1,2)={p0_owns_12}, armies={armies_12}")

# Wait a few turns to build armies
for i in range(5):
    state, _ = jax_step(state, jnp.array([[1,0,0,0,0], [1,0,0,0,0]], dtype=jnp.int32))

armies_12_now = int(np.array(state.armies[1, 2]))
print(f"  After 5 more turns: (1,2) has {armies_12_now} armies\n")

# THE CRITICAL TEST: Move from (1,2) LEFT back onto general at (1,1)
print("Step 2 (CRITICAL): P0 moves from (1,2) LEFT onto own general at (1,1)")
print("  This should NOT end the game!")

state_before = state
state, _ = jax_step(state, jnp.array([[0,1,2,3,0], [1,0,0,0,0]], dtype=jnp.int32))

winner = int(np.array(state.winner))
p0_still_owns_general = bool(np.array(state.ownership[0, 1, 1]))

print(f"\n  Result:")
print(f"    Winner: {winner}")
print(f"    P0 still owns general: {p0_still_owns_general}")

# CRITICAL ASSERTIONS
if winner >= 0:
    print(f"\n❌ BUG DETECTED: Game ended with winner={winner}")
    print("   Moving onto own general should NOT end the game!")
    exit(1)

if not p0_still_owns_general:
    print(f"\n❌ BUG DETECTED: P0 no longer owns general!")
    exit(1)

print(f"\n✅ TEST PASSED: Game did NOT end (winner={winner})")
print("✅ TEST PASSED: P0 still owns general")

print("\n" + "="*80)
print("🎉 SELF-CAPTURE BUG TEST PASSED!")
print("="*80)
print("\nThe fix is working correctly:")
print("  general_captured = attacker_wins & is_general & ~moving_to_own")
print("\nThis prevents players from 'capturing' their own generals.")
print("="*80 + "\n")
