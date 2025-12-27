"""
Clean JAX PPO using the env_jax.GeneralsEnv class.

Same simple structure as jax_ppo_2.py but using the environment abstraction.
"""

import sys
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core.env_jax import GeneralsEnv
from generals.core.action_jax import compute_valid_move_mask
from generals.core import game_jax
from generals.core.rewards_jax import composite_reward_fn


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

        # Larger backbone with more layers and channels
        x = jax.nn.relu(self.conv1(obs))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = jax.nn.relu(self.conv4(x))

        # Value head: more channels and an extra MLP layer
        v = jax.nn.relu(self.value_conv(x))
        v_flat = v.reshape(-1)
        value_hidden = jax.nn.relu(self.value_linear1(v_flat))
        value = self.value_linear2(value_hidden)[0]

        # Policy
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
            # Sample
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


@jax.jit
def rollout_step(states, network, key):
    """Vectorized rollout step for all environments."""
    num_envs = states.armies.shape[0]
    
    # Observations (BEFORE step for reward calculation)
    obs_p0_prior = jax.vmap(lambda s: game_jax.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game_jax.get_observation(s, 1))(states)
    
    # Actions from network
    obs_arr = jax.vmap(obs_to_array)(obs_p0_prior)
    masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0_prior)
    
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p0, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )
    
    # Random actions for p1
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p1 = jax.vmap(random_action)(jnp.stack(keys), obs_p1_prior)
    
    # Step game
    actions = jnp.stack([actions_p0, actions_p1], axis=1)
    new_states, infos = jax.vmap(game_jax.step)(states, actions)
    
    # Get new observations (AFTER step)
    obs_p0_new = jax.vmap(lambda s: game_jax.get_observation(s, 0))(new_states)
    
    # Compute rewards using composite reward function
    rewards = jax.vmap(composite_reward_fn)(
        obs_p0_prior, actions_p0, obs_p0_new
    )
    
    # Terminated/truncated
    terminated = infos.is_done
    truncated = (new_states.time >= 500) & ~terminated
    dones = terminated | truncated
    
    # Auto-reset if done
    grid = jnp.zeros((4, 4), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1).at[3, 3].set(2)
    grids = jnp.stack([grid] * num_envs)
    reset_states = jax.vmap(game_jax.create_initial_state)(grids)
    
    final_states = jax.tree.map(
        lambda reset, current: jnp.where(dones.reshape(num_envs, *([1] * (reset.ndim - 1))), reset, current),
        reset_states,
        new_states
    )
    
    return final_states, (obs_arr, masks, actions_p0, logprobs, values, rewards, dones, infos), key


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


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    num_steps = 256
    num_iterations = 200
    lr = 3e-4
    
    print(f"JAX PPO with GeneralsEnv - {num_envs} envs, {num_steps} steps/rollout")
    print(f"Device: {jax.devices()[0]}")
    print(f"Grid: 4x4 with composite rewards")
    print()
    
    # Initialize
    key = jrandom.PRNGKey(42)
    key, net_key = jrandom.split(key)
    network = TinyNetwork(net_key, grid_size=4)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    params, _ = eqx.partition(network, eqx.is_array)
    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    # Create environment (for documentation - actual logic is inlined for speed)
    env = GeneralsEnv(truncation=500)
    
    # Initialize states
    key, *reset_keys = jrandom.split(key, num_envs + 1)
    states = jax.vmap(env.reset)(jnp.stack(reset_keys))
    
    print("\nWarming up...")
    for _ in range(3):
        states, _, key = rollout_step(states, network, key)
    
    print("Training...\n")
    
    for iteration in range(num_iterations):
        t0 = time.time()
        
        # Collect rollout
        rollout_data = []
        for _ in range(num_steps):
            states, data, key = rollout_step(states, network, key)
            rollout_data.append(data)
        
        # Stack data
        obs = jnp.stack([d[0] for d in rollout_data])
        masks = jnp.stack([d[1] for d in rollout_data])
        actions = jnp.stack([d[2] for d in rollout_data])
        logprobs = jnp.stack([d[3] for d in rollout_data])
        values = jnp.stack([d[4] for d in rollout_data])
        rewards = jnp.stack([d[5] for d in rollout_data])
        dones = jnp.stack([d[6] for d in rollout_data])
        infos_list = [d[7] for d in rollout_data]
        infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos_list)
        
        # Compute advantages
        advantages = compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        # Train
        batch = (obs, masks, actions, logprobs, advantages, returns)
        network, opt_state, loss = train_step(network, opt_state, batch, optimizer)
        
        elapsed = time.time() - t0
        
        if iteration % 10 == 0:
            avg_reward = rewards.mean()
            num_episodes = int(dones.sum())
            wins = int(jnp.sum((dones) & (infos.winner == 0)))
            losses = int(jnp.sum((dones) & (infos.winner == 1)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (num_envs * num_steps) / elapsed
            print(f"Iter {iteration:4d} | Loss: {float(loss):.4f} | "
                  f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                  f"Wins: {wins:2d}/{num_episodes} ({win_rate:.0f}%) | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s")
    
    print("\nTraining complete!")
    
    # Save model
    model_path = "jax_ppo_model_200iters.eqx"
    eqx.tree_serialise_leaves(model_path, network)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
