"""
Visualize a trained JAX PPO policy on Generals.io.

Loads a saved model and runs episodes with GUI rendering to see how the policy plays.
"""

import sys
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from generals.core import game
from generals.core.action import compute_valid_move_mask


class TinyNetwork(eqx.Module):
    """Bigger network with value head for 4x4 grid."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    policy_conv: eqx.nn.Conv2d
    value_conv: eqx.nn.Conv2d
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear

    def __init__(self, key, grid_size=4):
        key1, key2, key3, key4, key5, key6 = jrandom.split(key, 6)

        self.conv1 = eqx.nn.Conv2d(9, 32, kernel_size=3, padding=1, key=key1)
        self.conv2 = eqx.nn.Conv2d(32, 32, kernel_size=3, padding=1, key=key2)
        self.conv3 = eqx.nn.Conv2d(32, 32, kernel_size=3, padding=1, key=key3)
        self.conv4 = eqx.nn.Conv2d(32, 16, kernel_size=3, padding=1, key=key4)
        self.policy_conv = eqx.nn.Conv2d(16, 9, kernel_size=1, key=key5)
        self.value_conv = eqx.nn.Conv2d(16, 4, kernel_size=1, key=key6)
        self.value_linear1 = eqx.nn.Linear(grid_size * grid_size * 4, 64, key=key5)
        self.value_linear2 = eqx.nn.Linear(64, 1, key=key6)

    def __call__(self, obs, mask, key, action=None):
        """Forward pass. Returns (action, value, logprob, entropy)."""
        grid_size = obs.shape[-1]

        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.conv4(x))

        v = jax.nn.relu(self.value_conv(x))
        v_flat = v.reshape(-1)
        value_hidden = jax.nn.relu(self.value_linear1(v_flat))
        value = self.value_linear2(value_hidden)[0]

        logits = self.policy_conv(x)
        mask_t = jnp.transpose(mask, (2, 0, 1))
        mask_penalty = (1 - mask_t) * -1e9
        combined_mask = jnp.concatenate([
            mask_penalty, mask_penalty, 
            jnp.zeros((1, grid_size, grid_size))
        ], axis=0)
        logits = (logits + combined_mask).reshape(-1)

        grid_cells = grid_size * grid_size

        if action is None:
            idx = jrandom.categorical(key, logits)
            direction, position = idx // grid_cells, idx % grid_cells
            row, col = position // grid_size, position % grid_size
            is_pass, is_half = direction == 8, (direction >= 4) & (direction < 8)
            actual_dir = jnp.where(is_pass, 0, jnp.where(is_half, direction - 4, direction))
            action = jnp.array([is_pass, row, col, actual_dir, is_half], dtype=jnp.int32)
        else:
            is_pass, row, col, direction, is_half = action[0], action[1], action[2], action[3], action[4]
            encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
            idx = encoded_dir * grid_cells + row * grid_size + col

        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[idx]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)

        return action, value, logprob, entropy


def obs_to_array(obs):
    """Convert observation to array."""
    return jnp.stack([
        obs.armies, obs.generals, obs.cities, obs.mountains,
        obs.neutral_cells, obs.owned_cells, obs.opponent_cells,
        obs.fog_cells, obs.structures_in_fog
    ], axis=0).astype(jnp.float32)


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


def visualize_episode(network, grid, key, episode_num, gui):
    """Run a single episode with GUI rendering."""
    state = game.create_initial_state(grid)
    done = False
    step_count = 0
    max_steps = 500
    
    print(f"\n{'='*80}")
    print(f"Episode {episode_num}")
    print('='*80)
    
    while not done and step_count < max_steps:
        obs_p0 = game.get_observation(state, 0)
        obs_p1 = game.get_observation(state, 1)
        
        # Player 0 uses trained policy
        obs_arr = obs_to_array(obs_p0)
        mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
        
        key, subkey = jrandom.split(key)
        action_p0, value, _, _ = network(obs_arr, mask, subkey, None)
        
        # Player 1 uses random policy
        key, subkey = jrandom.split(key)
        action_p1 = random_action(subkey, obs_p1)
        
        # Take step
        actions = jnp.stack([action_p0, action_p1])
        state, info = game.step(state, actions)
        
        done = bool(info.is_done)
        step_count += 1
        
        # Update GUI with new state
        gui.properties.game.update_from_state(state, info)
        
        try:
            command = gui.tick(fps=30)
            if command.quit:
                return key, "QUIT"
        except SystemExit:
            return key, "QUIT"
    
    # Final result
    if info.winner == 0:
        result = "WIN"
    elif info.winner == 1:
        result = "LOSS"
    else:
        result = "DRAW"
    
    p0_cells = int(obs_p0.owned_cells.sum())
    p1_cells = int(obs_p1.owned_cells.sum())
    
    print(f"{'='*80}")
    print(f"Result: {result} after {step_count} steps")
    print(f"Final score - Player 0: {p0_cells} cells, Player 1: {p1_cells} cells")
    print('='*80)
    
    return key, result


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "jax_ppo_model_200iters.eqx"
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    grid_size = 4
    
    print("="*80)
    print("JAX PPO Policy Visualization")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes to run: {num_episodes}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print("\nControls:")
    print("  - ESC or close window to quit")
    print("  - Watch the PPO agent (red) vs random agent (blue)")
    print("="*80)
    
    # Initialize network structure
    key = jrandom.PRNGKey(0)
    key, net_key = jrandom.split(key)
    network = TinyNetwork(net_key, grid_size=grid_size)
    
    # Load trained weights
    print(f"\nLoading model from {model_path}...")
    network = eqx.tree_deserialise_leaves(model_path, network)
    print("Model loaded successfully!")
    
    # Setup grid
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[grid_size-1, grid_size-1].set(2)
    
    # Setup GUI
    from generals.envs.jax_rendering_adapter import JaxGameAdapter
    from generals.gui import GUI
    from generals.gui.properties import GuiMode
    
    agent_names = ['PPO Agent', 'Random Agent']
    initial_state = game.create_initial_state(grid)
    initial_info = game.get_info(initial_state)
    game_adapter = JaxGameAdapter(initial_state, agent_names, initial_info)
    
    agent_data = {
        'PPO Agent': {'color': (255, 50, 50)},  # Red
        'Random Agent': {'color': (50, 50, 255)}  # Blue
    }
    
    gui = GUI(game_adapter, agent_data, mode=GuiMode.TRAIN, speed_multiplier=0.5)
    
    # Run episodes
    results = {"WIN": 0, "LOSS": 0, "DRAW": 0, "QUIT": 0}
    
    try:
        for episode in range(1, num_episodes + 1):
            key, result = visualize_episode(network, grid, key, episode, gui)
            results[result] += 1
            
            if result == "QUIT":
                print("\nVisualization stopped by user")
                break
    finally:
        gui.close()
    
    # Summary
    if results["QUIT"] == 0:
        total_episodes = num_episodes
    else:
        total_episodes = sum(results.values()) - 1
    
    if total_episodes > 0:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total episodes: {total_episodes}")
        print(f"Wins:   {results['WIN']:3d} ({results['WIN']/total_episodes*100:5.1f}%)")
        print(f"Losses: {results['LOSS']:3d} ({results['LOSS']/total_episodes*100:5.1f}%)")
        print(f"Draws:  {results['DRAW']:3d} ({results['DRAW']/total_episodes*100:5.1f}%)")
        print("="*80)


if __name__ == "__main__":
    main()
