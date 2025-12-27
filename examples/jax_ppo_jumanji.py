"""
Clean JAX PPO using Jumanji-style GeneralsEnv.

Fully vectorized, no loops, maximum performance.
"""

import sys
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core.env_jax import GeneralsEnv, make_env
from generals.core.action_jax import compute_valid_move_mask
from generals.core import game_jax


class PolicyValueNetwork(eqx.Module):
    """CNN policy-value network."""

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
    """Convert ObservationJax to array for network input."""
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


def rollout_step(state, key, network, reward_fn, grid_factory):
    """Single rollout step (P0 network, P1 random)."""
    # Get observations
    obs_p0 = game_jax.get_observation(state, 0)
    obs_p1 = game_jax.get_observation(state, 1)
    
    # P0 uses network
    obs_arr = obs_to_array(obs_p0)
    mask = compute_valid_move_mask(obs_p0.armies, obs_p0.owned_cells, obs_p0.mountains)
    
    key, k1, k2, k3 = jrandom.split(key, 4)
    action_p0, value, logprob, entropy = network(obs_arr, mask, k1, None)
    
    # P1 random
    action_p1 = random_action(k2, obs_p1)
    
    # Step environment manually (inline the env.step logic)
    actions = jnp.stack([action_p0, action_p1])
    
    # Get prior observations for reward
    obs_p0_prior = obs_p0
    
    # Game step
    new_state, info = game_jax.step(state, actions)
    
    # Get new observation
    obs_p0_new = game_jax.get_observation(new_state, 0)
    
    # Compute reward
    reward = reward_fn(obs_p0_prior, action_p0, obs_p0_new)
    
    # Auto-reset if done
    reset_grid = grid_factory(k3)
    reset_state = game_jax.create_initial_state(reset_grid)
    final_state = jax.tree.map(
        lambda reset, current: jnp.where(info.is_done, reset, current),
        reset_state,
        new_state
    )
    
    return final_state, (obs_arr, mask, action_p0, value, logprob, reward, info.is_done), key


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute advantages using GAE."""
    num_steps = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    last_adv = 0.0
    
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


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    num_steps = 256
    num_iterations = 200
    lr = 3e-4
    grid_size = 4
    
    print("="*80)
    print("JAX PPO with Jumanji-style Environment")
    print("="*80)
    print(f"Device: {jax.devices()[0]}")
    print(f"Envs: {num_envs} | Steps/rollout: {num_steps} | Iterations: {num_iterations}")
    print(f"Grid: {grid_size}x{grid_size}")
    print(f"Reward: composite_reward_fn")
    print("="*80)
    print()
    
    # Create environment
    env = make_env(reward_fn_name='composite', grid_size=grid_size)
    
    # Vectorize environment functions
    batch_reset = jax.jit(jax.vmap(env.reset))
    batch_step = jax.jit(jax.vmap(env.step))
    
    # Initialize network
    key = jrandom.PRNGKey(42)
    key, net_key = jrandom.split(key)
    network = PolicyValueNetwork(net_key, grid_size=grid_size)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    params, _ = eqx.partition(network, eqx.is_array)
    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    # Initialize states
    key, *reset_keys = jrandom.split(key, num_envs + 1)
    states = batch_reset(jnp.stack(reset_keys))
    
    # Bind reward_fn and grid_factory for JIT
    from functools import partial
    bound_rollout_step = jax.jit(partial(rollout_step, network=network, reward_fn=env.reward_fn, grid_factory=env.grid_factory))
    
    print("\nWarming up JIT compilation...")
    for _ in range(3):
        key, *step_keys = jrandom.split(key, num_envs + 1)
        states, _, _ = jax.vmap(bound_rollout_step)(states, jnp.stack(step_keys))
    
    print("Training...\n")
    
    # JIT compile the entire rollout collection
    @jax.jit
    def collect_rollout(states, key):
        """Collect full rollout using lax.scan for speed."""
        def scan_fn(carry, _):
            state, k = carry
            k, *step_keys = jrandom.split(k, num_envs + 1)
            new_states, data, _ = jax.vmap(bound_rollout_step)(state, jnp.stack(step_keys))
            return (new_states, k), data
        
        (final_states, final_key), rollout_data = jax.lax.scan(scan_fn, (states, key), None, length=num_steps)
        return final_states, rollout_data, final_key
    
    for iteration in range(num_iterations):
        t0 = time.time()
        
        # Collect rollout (fully JIT-compiled)
        states, rollout_data, key = collect_rollout(states, key)
        
        # Unpack rollout data
        obs, masks, actions, values, logprobs, rewards, dones = rollout_data
        
        # Compute advantages (per environment)
        advantages_list = []
        for env_idx in range(num_envs):
            adv = compute_gae(rewards[:, env_idx], values[:, env_idx], dones[:, env_idx])
            advantages_list.append(adv)
        advantages = jnp.stack(advantages_list, axis=1)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        # Train
        batch = (obs, masks, actions, logprobs, advantages, returns)
        network, opt_state, loss = train_step(network, opt_state, batch, optimizer)
        
        elapsed = time.time() - t0
        
        if iteration % 10 == 0:
            avg_reward = rewards.mean()
            num_episodes = int(dones.sum())
            sps = (num_envs * num_steps) / elapsed
            print(f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                  f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s")
    
    print("\n" + "="*80)
    print("Training complete! Saving model...")
    print("="*80)
    
    # Save model
    model_path = "jax_ppo_model_200iters.eqx"
    eqx.tree_serialise_leaves(model_path, network)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
