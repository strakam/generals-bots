"""
Visualize a trained PPO policy playing against a random opponent.

Loads a saved model and renders the game using pygame.
"""

import sys
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import pygame

from generals.core.action import compute_valid_move_mask
from generals.core import game
from generals.core.game import GameState, GameInfo
from generals.gui import GUI
from generals.gui.properties import GuiMode
from generals.envs.jax_rendering_adapter import JaxGameAdapter

try:
    from ppo.network import PolicyValueNetwork, obs_to_array
except ImportError:
    from examples.ppo.network import PolicyValueNetwork, obs_to_array


def random_action(key, obs):
    """Random valid action."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = jnp.argwhere(mask, size=100, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid >= 0, axis=-1))
    
    k1, k2 = jrandom.split(key)
    should_pass = num_valid == 0
    idx = jnp.minimum(jrandom.randint(k1, (), 0, jnp.maximum(num_valid, 1)), num_valid - 1)
    move = valid[idx]
    is_half = jrandom.randint(k2, (), 0, 2)
    
    return jnp.array([should_pass, move[0], move[1], move[2], is_half], dtype=jnp.int32)


def load_model(model_path: str, grid_size: int = 4):
    """Load a trained PPO model from file."""
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create a dummy network with the same structure
    key = jrandom.PRNGKey(42)
    network = PolicyValueNetwork(key, grid_size=grid_size)
    
    # Load the saved weights
    network = eqx.tree_deserialise_leaves(model_path, network)
    print(f"Loaded model from {model_path}")
    return network


def main():
    # Parse command line arguments
    model_path = sys.argv[1] if len(sys.argv) > 1 else "jax_ppo_model.eqx"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Loading model from: {model_path}")
    print(f"Rendering at {fps} FPS")
    print()
    
    # Load the trained model
    network = load_model(model_path, grid_size=4)
    
    # Initialize game state
    key = jrandom.PRNGKey(42)
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[3, 3].set(2)
    state = game.create_initial_state(grid)
    
    # Agent names
    agents = ["PPO Agent", "Random"]
    
    # Create game adapter for rendering
    info = game.get_info(state)
    game_adapter = JaxGameAdapter(state, agents, info)
    
    # Create agent data for GUI
    agent_data = {
        "PPO Agent": {"color": (255, 0, 0)},  # Red
        "Random": {"color": (0, 0, 255)},      # Blue
    }
    
    # Initialize GUI
    gui = GUI(game_adapter, agent_data, mode=GuiMode.TRAIN, speed_multiplier=1.0)
    
    print("Starting game visualization...")
    print("Controls:")
    print("  - Close window to exit")
    print()
    
    # Game loop
    step_count = 0
    max_steps = 500
    clock = pygame.time.Clock()
    
    while step_count < max_steps:
        # Handle pygame events
        command = gui.tick(fps)
        if command.quit:
            break
        
        # Check if game is done
        info = game.get_info(state)
        if info.is_done:
            winner_idx = int(info.winner)
            winner_name = agents[winner_idx] if winner_idx >= 0 else "Draw"
            print(f"\nGame over! Winner: {winner_name}")
            print(f"Total steps: {step_count}")
            
            # Wait a bit before resetting
            time.sleep(2)
            
            # Reset game
            key, reset_key = jrandom.split(key)
            grid = jnp.zeros((4, 4), dtype=jnp.int32)
            grid = grid.at[0, 0].set(1).at[3, 3].set(2)
            state = game.create_initial_state(grid)
            info = game.get_info(state)
            game_adapter.update_from_state(state, info)
            step_count = 0
            print("Starting new game...")
            continue
        
        # Get observations
        obs_p0 = game.get_observation(state, 0)
        obs_p1 = game.get_observation(state, 1)
        
        # PPO agent action (player 0)
        obs_arr = obs_to_array(obs_p0)
        mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
        key, action_key = jrandom.split(key)
        action_p0, value, logprob, entropy = network(obs_arr, mask, action_key, None)
        
        # Random agent action (player 1)
        key, action_key = jrandom.split(key)
        action_p1 = random_action(action_key, obs_p1)
        
        # Step the game
        actions = jnp.stack([action_p0, action_p1], axis=0)
        new_state, new_info = game.step(state, actions)
        
        # Update state
        state = new_state
        info = new_info
        
        # Update GUI adapter
        game_adapter.update_from_state(state, info)
        
        step_count += 1
        
        # Control frame rate
        clock.tick(fps)
    
    gui.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

