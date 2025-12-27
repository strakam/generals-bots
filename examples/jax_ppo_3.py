"""
Clean JAX PPO implementation following Gymnasium pattern.

Key improvements:
- Proper separation: env steps, agent acts, rewards computed separately
- Uses rewards_jax from core (composite_reward_fn, etc.)
- Pluggable reward functions
- Clean Gymnasium-style interface
- Saves model after 200 iterations and visualizes
"""

import sys
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core import game_jax
from generals.core.action_jax import compute_valid_move_mask
from generals.core.rewards_jax import composite_reward_fn


class TinyNetwork(eqx.Module):
    """CNN policy-value network for 4x4 grid."""

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

        # Value head
        v = jax.nn.relu(self.value_conv(x))
        v_flat = v.reshape(-1)
        value_hidden = jax.nn.relu(self.value_linear1(v_flat))
        value = self.value_linear2(value_hidden)[0]

        # Policy head
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
            # Sample action
            idx = jrandom.categorical(key, logits)
            direction, position = idx // grid_cells, idx % grid_cells
            row, col = position // grid_size, position % grid_size
            is_pass, is_half = direction == 8, (direction >= 4) & (direction < 8)
            actual_dir = jnp.where(is_pass, 0, jnp.where(is_half, direction - 4, direction))
            action = jnp.array([is_pass, row, col, actual_dir, is_half], dtype=jnp.int32)
        else:
            # Compute index from action
            is_pass, row, col, direction, is_half = action[0], action[1], action[2], action[3], action[4]
            encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
            idx = encoded_dir * grid_cells + row * grid_size + col

        # Log prob and entropy
        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[idx]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)

        return action, value, logprob, entropy


def obs_to_array(obs):
    """Convert ObservationJax to array for network input."""
    return jnp.stack([
        obs.armies, obs.generals, obs.cities, obs.mountains,
        obs.neutral_cells, obs.owned_cells, obs.opponent_cells,
        obs.fog_cells, obs.structures_in_fog
    ], axis=0).astype(jnp.float32)


def random_action(key, obs):
    """Random valid action for opponent."""
    mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    valid = jnp.argwhere(mask, size=100, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid >= 0, axis=-1))
    
    k1, k2 = jrandom.split(key)
    should_pass = num_valid == 0
    idx = jnp.minimum(jrandom.randint(k1, (), 0, jnp.maximum(num_valid, 1)), num_valid - 1)
    move = valid[idx]
    is_half = jrandom.randint(k2, (), 0, 2)
    
    return jnp.array([should_pass, move[0], move[1], move[2], is_half], dtype=jnp.int32)


@jax.jit
def env_step(states, actions, grids):
    """
    Vectorized environment step (Gymnasium pattern).
    
    Args:
        states: [num_envs] GameState
        actions: [num_envs, 2, 5] actions for both players
        grids: [num_envs, H, W] grids for reset
        
    Returns:
        observations: [num_envs, 2, ...] observations for both players
        rewards: [num_envs, 2] rewards for both players
        terminated: [num_envs] done flags
        states: [num_envs] new states (auto-reset if done)
    """
    num_envs = states.armies.shape[0]
    
    # Get prior observations (before step)
    obs_p0_prior = jax.vmap(lambda s: game_jax.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game_jax.get_observation(s, 1))(states)
    
    # Step environment
    new_states, infos = jax.vmap(game_jax.step)(states, actions)
    
    # Get new observations (after step)
    obs_p0 = jax.vmap(lambda s: game_jax.get_observation(s, 0))(new_states)
    obs_p1 = jax.vmap(lambda s: game_jax.get_observation(s, 1))(new_states)
    
    # Compute rewards using composite_reward_fn (prior_obs, action, new_obs)
    rewards_p0 = jax.vmap(composite_reward_fn)(obs_p0_prior, actions[:, 0], obs_p0)
    rewards_p1 = jax.vmap(composite_reward_fn)(obs_p1_prior, actions[:, 1], obs_p1)
    rewards = jnp.stack([rewards_p0, rewards_p1], axis=1)
    
    # Done flags
    terminated = infos.is_done
    
    # Auto-reset done environments
    fresh = jax.vmap(game_jax.create_initial_state)(grids)
    states = jax.tree.map(
        lambda f, c: jnp.where(terminated.reshape(num_envs, *([1] * (f.ndim - 1))), f, c),
        fresh, new_states
    )
    
    # Stack observations for both players
    observations = jax.tree.map(
        lambda p0, p1: jnp.stack([p0, p1], axis=1),
        obs_p0, obs_p1
    )
    
    return observations, rewards, terminated, states, infos


def sample_actions_batch(network, observations, key):
    """
    Sample actions for both players (P0 uses network, P1 random).
    
    Args:
        network: Policy network
        observations: Batched observations [num_envs, 2, ...]
        key: JAX random key
        
    Returns:
        actions: [num_envs, 2, 5] actions
        values: [num_envs] value estimates for P0
        logprobs: [num_envs] log probabilities for P0
        entropies: [num_envs] entropies for P0
    """
    num_envs = observations.armies.shape[0]
    
    # Extract player 0 observations
    obs_p0 = jax.tree.map(lambda x: x[:, 0], observations)
    obs_p1 = jax.tree.map(lambda x: x[:, 1], observations)
    
    # Convert to arrays and get masks
    obs_arr = jax.vmap(obs_to_array)(obs_p0)
    masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
    
    # Sample actions for P0 using network
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p0, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )
    
    # Random actions for P1
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p1 = jax.vmap(random_action)(jnp.stack(keys), obs_p1)
    
    # Stack actions
    actions = jnp.stack([actions_p0, actions_p1], axis=1)
    
    return actions, values, logprobs, entropies, key


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute advantages using GAE."""
    num_steps, num_envs = rewards.shape
    advantages = jnp.zeros_like(rewards)
    last_adv = jnp.zeros(num_envs)
    
    for t in reversed(range(num_steps)):
        next_value = jnp.where(t == num_steps - 1, 0.0, values[t + 1])
        next_nonterminal = jnp.where(t == num_steps - 1, 1.0 - dones[t], 1.0 - dones[t + 1])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        advantages = advantages.at[t].set(delta + gamma * lam * next_nonterminal * last_adv)
        last_adv = advantages[t]
    
    return advantages


@jax.jit
def ppo_loss(network, obs, mask, action, old_logprob, advantage, ret, clip=0.2):
    """PPO loss for single sample."""
    _, value, logprob, entropy = network(obs, mask, None, action)
    
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
    policy_loss = -jnp.minimum(ratio * advantage, clipped)
    
    value_loss = 0.5 * (value - ret) ** 2
    entropy_loss = -0.01 * entropy
    
    return policy_loss + value_loss + entropy_loss


def train_step(network, opt_state, batch, optimizer):
    """Single training step."""
    obs, masks, actions, old_logprobs, advantages, returns = batch
    
    # Flatten batch
    bs = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(bs, *obs.shape[2:])
    masks_flat = masks.reshape(bs, *masks.shape[2:])
    actions_flat = actions.reshape(bs, -1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)
    
    def loss_fn(net):
        losses = jax.vmap(lambda o, m, a, olp, adv, r: ppo_loss(net, o, m, a, olp, adv, r))(
            obs_flat, masks_flat, actions_flat, old_logprobs_flat, advantages_flat, returns_flat
        )
        return jnp.mean(losses)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    
    return network, opt_state, loss


def visualize_policy(network, num_episodes=20):
    """Visualize trained policy with GUI rendering."""
    from generals.envs.jax_rendering_adapter import JaxGameAdapter
    from generals.gui import GUI
    from generals.gui.properties import GuiMode
    from generals.core import game_jax
    
    print("\n" + "="*80)
    print("Visualizing policy with GUI rendering...")
    print("="*80)
    
    # Create single environment for visualization
    grid_size = 4
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[grid_size-1, grid_size-1].set(2)
    
    agent_names = ['PPO Agent', 'Random Agent']
    initial_state = game_jax.create_initial_state(grid)
    initial_info = game_jax.get_info(initial_state)
    game_adapter = JaxGameAdapter(initial_state, agent_names, initial_info)
    
    agent_data = {
        'PPO Agent': {'color': (255, 50, 50)},
        'Random Agent': {'color': (50, 50, 255)}
    }
    
    gui = GUI(game_adapter, agent_data, mode=GuiMode.TRAIN, speed_multiplier=0.5)
    
    key = jrandom.PRNGKey(42)
    results = {"WIN": 0, "LOSS": 0, "DRAW": 0, "QUIT": 0}
    
    try:
        for episode in range(1, num_episodes + 1):
            state = game_jax.create_initial_state(grid)
            done = False
            step_count = 0
            max_steps = 500
            
            print(f"\n=== Episode {episode} ===")
            
            while not done and step_count < max_steps:
                obs_p0 = game_jax.get_observation(state, 0)
                obs_p1 = game_jax.get_observation(state, 1)
                
                # P0 uses trained policy
                obs_arr = obs_to_array(obs_p0)
                mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
                
                key, subkey = jrandom.split(key)
                action_p0, _, _, _ = network(obs_arr, mask, subkey, None)
                
                # P1 random
                key, subkey = jrandom.split(key)
                action_p1 = random_action(subkey, obs_p1)
                
                # Step
                actions = jnp.stack([action_p0, action_p1])
                state, info = game_jax.step(state, actions)
                
                done = bool(info.is_done)
                step_count += 1
                
                # Update GUI
                gui.properties.game.update_from_state(state, info)
                
                try:
                    command = gui.tick(fps=30)
                    if command.quit:
                        results["QUIT"] += 1
                        raise KeyboardInterrupt
                except SystemExit:
                    results["QUIT"] += 1
                    raise KeyboardInterrupt
            
            # Record result
            if info.winner == 0:
                results["WIN"] += 1
            elif info.winner == 1:
                results["LOSS"] += 1
            else:
                results["DRAW"] += 1
                
            print(f"Result: {'WIN' if info.winner == 0 else 'LOSS' if info.winner == 1 else 'DRAW'} after {step_count} steps")
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        gui.close()
    
    # Summary
    total = results["WIN"] + results["LOSS"] + results["DRAW"]
    if total > 0:
        print("\n" + "="*80)
        print("VISUALIZATION SUMMARY")
        print("="*80)
        print(f"Total episodes: {total}")
        print(f"Wins:   {results['WIN']:2d} ({results['WIN']/total*100:.0f}%)")
        print(f"Losses: {results['LOSS']:2d} ({results['LOSS']/total*100:.0f}%)")
        print(f"Draws:  {results['DRAW']:2d} ({results['DRAW']/total*100:.0f}%)")
        print("="*80)


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    num_steps = 256
    num_iterations = 200
    lr = 3e-4
    grid_size = 4
    
    print(f"JAX PPO with Clean Gymnasium Pattern")
    print(f"Device: {jax.devices()[0]}")
    print(f"Envs: {num_envs} | Steps/rollout: {num_steps} | Iterations: {num_iterations}")
    print(f"Reward: composite_reward_fn from core.rewards_jax")
    print(f"Grid: {grid_size}x{grid_size}")
    print()
    
    # Initialize network
    key = jrandom.PRNGKey(42)
    key, net_key = jrandom.split(key)
    network = TinyNetwork(net_key, grid_size=grid_size)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    params, _ = eqx.partition(network, eqx.is_array)
    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    # Setup grids and initial states
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[grid_size-1, grid_size-1].set(2)
    grids = jnp.stack([grid] * num_envs)
    states = jax.vmap(game_jax.create_initial_state)(grids)
    
    print("\nWarming up JIT compilation...")
    for _ in range(3):
        obs_p0 = jax.vmap(lambda s: game_jax.get_observation(s, 0))(states)
        obs_p1 = jax.vmap(lambda s: game_jax.get_observation(s, 1))(states)
        observations = jax.tree.map(lambda p0, p1: jnp.stack([p0, p1], axis=1), obs_p0, obs_p1)
        
        actions, _, _, _, key = sample_actions_batch(network, observations, key)
        observations, rewards, terminated, states, infos = env_step(states, actions, grids)
    
    print("Training...\n")
    
    for iteration in range(num_iterations):
        t0 = time.time()
        
        print(f"Iteration {iteration} starting...", flush=True)
        
        # Collect rollout
        rollout_data = []
        
        for step_idx in range(num_steps):
            if iteration <= 1 and step_idx == 0:
                print(f"  Rollout step {step_idx}...", flush=True)
            # Get observations
            obs_p0 = jax.vmap(lambda s: game_jax.get_observation(s, 0))(states)
            obs_p1 = jax.vmap(lambda s: game_jax.get_observation(s, 1))(states)
            observations = jax.tree.map(lambda p0, p1: jnp.stack([p0, p1], axis=1), obs_p0, obs_p1)
            
            # Sample actions
            actions, values, logprobs, entropies, key = sample_actions_batch(network, observations, key)
            
            # Environment step
            next_observations, rewards, terminated, states, infos = env_step(states, actions, grids)
            
            # Extract P0 data for training
            obs_arr = jax.vmap(obs_to_array)(obs_p0)
            masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0)
            actions_p0 = actions[:, 0]
            rewards_p0 = rewards[:, 0]
            
            rollout_data.append((obs_arr, masks, actions_p0, logprobs, values, rewards_p0, terminated, infos))
        
        # Stack rollout data
        obs_stack = jnp.stack([d[0] for d in rollout_data])
        masks_stack = jnp.stack([d[1] for d in rollout_data])
        actions_stack = jnp.stack([d[2] for d in rollout_data])
        logprobs_stack = jnp.stack([d[3] for d in rollout_data])
        values_stack = jnp.stack([d[4] for d in rollout_data])
        rewards_stack = jnp.stack([d[5] for d in rollout_data])
        dones_stack = jnp.stack([d[6] for d in rollout_data])
        infos_list = [d[7] for d in rollout_data]
        infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos_list)
        
        # Compute advantages
        advantages = compute_gae(rewards_stack, values_stack, dones_stack)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values_stack
        
        # Train
        batch = (obs_stack, masks_stack, actions_stack, logprobs_stack, advantages, returns)
        if iteration % 10 == 0:
            print(f"  Training on batch...", flush=True)
        network, opt_state, loss = train_step(network, opt_state, batch, optimizer)
        if iteration % 10 == 0:
            print(f"  Training done. Loss: {float(loss):.4f}", flush=True)
        
        elapsed = time.time() - t0
        
        if iteration % 10 == 0:
            avg_reward = rewards_stack.mean()
            num_episodes = int(dones_stack.sum())
            wins = int(jnp.sum(dones_stack & (infos.winner == 0)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (num_envs * num_steps) / elapsed
            print(f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                  f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                  f"Wins: {wins:2d}/{num_episodes} ({win_rate:.0f}%) | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s", flush=True)
    
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    print("="*80)
    
    # Save model
    model_path = "jax_ppo_model_200iters.eqx"
    eqx.tree_serialise_leaves(model_path, network)
    print(f"Model saved to: {model_path}")
    
    # Visualize
    visualize_policy(network, num_episodes=20)


if __name__ == "__main__":
    main()
