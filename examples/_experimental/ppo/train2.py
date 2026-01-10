"""
JAX PPO using GeneralsEnv wrapper from env.py.

This version uses the cleaner Gymnasium-like API for better code organization.
For maximum performance with raw game API, see examples/ppo/train.py
"""

import sys
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax

from generals.core.action import compute_valid_move_mask
from generals.core.env import GeneralsEnv
from generals.core.rewards import composite_reward_fn
from generals.core import game

from network import PolicyValueNetwork, obs_to_array


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


def _rollout_step_inner(states, env, network, key):
    """Inner rollout step (not jitted, will be used inside scan)."""
    num_envs = states.armies.shape[0]
    
    # Get observations BEFORE step (for reward calculation)
    obs_p0_prior = jax.vmap(lambda s: game.get_observation(s, 0))(states)
    obs_p1_prior = jax.vmap(lambda s: game.get_observation(s, 1))(states)
    
    # Actions from network for p0
    obs_arr = jax.vmap(obs_to_array)(obs_p0_prior)
    masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(obs_p0_prior)
    
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p0, values, logprobs, entropies = jax.vmap(network, in_axes=(0, 0, 0, None))(
        obs_arr, masks, jnp.stack(keys), None
    )
    
    # Random actions for p1
    key, *keys = jrandom.split(key, num_envs + 1)
    actions_p1 = jax.vmap(random_action)(jnp.stack(keys), obs_p1_prior)
    
    # Stack actions for both players
    actions = jnp.stack([actions_p0, actions_p1], axis=1)
    
    # Step all environments (vmap over states, actions, and keys)
    key, *reset_keys = jrandom.split(key, num_envs + 1)
    timesteps, new_states = jax.vmap(lambda s, a, k: env.step(s, a, k))(
        states, actions, jnp.stack(reset_keys)
    )
    
    # Get new observations from timestep (BEFORE any auto-reset)
    # timestep.observation is [num_envs, 2, ...] for both players, extract player 0
    obs_p0_new = jax.tree.map(lambda x: x[:, 0], timesteps.observation)
    
    # Compute rewards using composite reward function (more detailed than env's win/lose)
    rewards = jax.vmap(composite_reward_fn)(
        obs_p0_prior, actions_p0, obs_p0_new
    )
    
    # Terminated/truncated from timestep
    dones = timesteps.terminated | timesteps.truncated
    
    return new_states, (obs_arr, masks, actions_p0, logprobs, values, rewards, dones, timesteps.info), key


@jax.jit(static_argnames=['env'])
def rollout_step(states, env, network, key):
    """Vectorized rollout step using GeneralsEnv (for warmup/single steps)."""
    return _rollout_step_inner(states, env, network, key)


def make_collect_rollout(env, num_steps):
    """Create a jitted rollout collection function using lax.scan."""
    
    def rollout_body(carry, _):
        states, network, key = carry
        new_states, data, new_key = _rollout_step_inner(states, env, network, key)
        return (new_states, network, new_key), data
    
    @jax.jit
    def collect_rollout(states, network, key):
        """Collect num_steps of rollout data using lax.scan."""
        (final_states, _, final_key), rollout_data = jax.lax.scan(
            rollout_body,
            (states, network, key),
            None,
            length=num_steps
        )
        return final_states, rollout_data, final_key
    
    return collect_rollout


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute advantages using GAE with lax.scan for efficient compilation."""
    num_steps, num_envs = rewards.shape
    
    # Append bootstrap value (0) for last step
    values_with_bootstrap = jnp.concatenate([values, jnp.zeros((1, num_envs))], axis=0)
    
    def gae_step(carry, inputs):
        last_adv = carry
        reward, value, next_value, done = inputs
        # Non-terminal mask for current step
        nonterminal = 1.0 - done
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * lam * nonterminal * last_adv
        return advantage, advantage
    
    # Prepare inputs for scan (process in reverse order)
    rewards_rev = rewards[::-1]
    values_rev = values[::-1]
    next_values_rev = values_with_bootstrap[1:][::-1]
    dones_rev = dones[::-1]
    
    inputs = (rewards_rev, values_rev, next_values_rev, dones_rev)
    init_adv = jnp.zeros(num_envs)
    
    _, advantages_rev = jax.lax.scan(gae_step, init_adv, inputs)
    
    # Reverse back to original order
    advantages = advantages_rev[::-1]
    
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


def train_step(network, opt_state, minibatch, optimizer):
    """Single training step on a minibatch."""
    obs, masks, actions, old_logprobs, advantages, returns = minibatch
    
    def loss_fn(net):
        losses = jax.vmap(lambda o, m, a, olp, adv, r: ppo_loss(net, o, m, a, olp, adv, r))(
            obs, masks, actions, old_logprobs, advantages, returns
        )
        return jnp.mean(losses)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(network)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    
    return network, opt_state, loss


def train_epoch(network, opt_state, batch, optimizer, key, minibatch_size):
    """Train for one epoch with minibatching (drops last incomplete batch to avoid recompilation)."""
    obs, masks, actions, old_logprobs, advantages, returns = batch
    
    # Flatten batch
    bs = obs.shape[0] * obs.shape[1]
    obs_flat = obs.reshape(bs, *obs.shape[2:])
    masks_flat = masks.reshape(bs, *masks.shape[2:])
    actions_flat = actions.reshape(bs, -1)
    old_logprobs_flat = old_logprobs.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)
    
    # Shuffle data
    perm = jrandom.permutation(key, bs)
    obs_flat = obs_flat[perm]
    masks_flat = masks_flat[perm]
    actions_flat = actions_flat[perm]
    old_logprobs_flat = old_logprobs_flat[perm]
    advantages_flat = advantages_flat[perm]
    returns_flat = returns_flat[perm]
    
    # Train on minibatches (drop last incomplete batch to avoid recompilation)
    num_complete_batches = bs // minibatch_size
    total_loss = 0.0
    
    for i in range(num_complete_batches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        minibatch = (
            obs_flat[start_idx:end_idx],
            masks_flat[start_idx:end_idx],
            actions_flat[start_idx:end_idx],
            old_logprobs_flat[start_idx:end_idx],
            advantages_flat[start_idx:end_idx],
            returns_flat[start_idx:end_idx],
        )
        network, opt_state, loss = train_step(network, opt_state, minibatch, optimizer)
        total_loss += loss
    
    avg_loss = total_loss / max(num_complete_batches, 1)
    return network, opt_state, avg_loss


def main():
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    num_steps = 300
    num_iterations = 500
    num_epochs = 1  # Multiple passes over collected data
    minibatch_size = 256  # Size of minibatches
    lr = 3e-4
    grid_size = 4  # Must match network architecture
    
    print(f"JAX PPO (GeneralsEnv API)")
    print(f"Environments:  {num_envs}")
    print(f"Device:        {jax.devices()[0]}")
    print(f"Grid:          {grid_size}x{grid_size} with composite rewards")
    print(f"Epochs:        {num_epochs}")
    print(f"Minibatch:     {minibatch_size}")
    print()
    
    # Initialize
    key = jrandom.PRNGKey(42)
    key, net_key = jrandom.split(key)
    network = PolicyValueNetwork(net_key, grid_size=grid_size)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    params, _ = eqx.partition(network, eqx.is_array)
    print(f"Parameters: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    # Create single environment instance with matching grid size
    env = GeneralsEnv(grid_dims=(grid_size, grid_size), truncation=500)
    
    # Create scan-based rollout collector
    collect_rollout = make_collect_rollout(env, num_steps)
    
    # Initialize states (vmap over reset keys)
    key, *reset_keys = jrandom.split(key, num_envs + 1)
    states = jax.vmap(env.reset)(jnp.stack(reset_keys))
    
    print("\nWarming up...")
    for _ in range(3):
        states, _, key = rollout_step(states, env, network, key)
    jax.block_until_ready(states)
    
    print("Training...\n")
    
    for iteration in range(num_iterations):
        t0 = time.time()
        
        # Collect rollout using lax.scan (more efficient than Python loop)
        states, rollout_data, key = collect_rollout(states, network, key)
        jax.block_until_ready(states)
        
        # Unpack rollout data (already stacked by scan)
        obs, masks, actions, logprobs, values, rewards, dones, infos = rollout_data
        
        # Compute advantages
        advantages = compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        # Train for multiple epochs
        batch = (obs, masks, actions, logprobs, advantages, returns)
        epoch_losses = []
        for epoch in range(num_epochs):
            key, epoch_key = jrandom.split(key)
            network, opt_state, loss = train_epoch(network, opt_state, batch, optimizer, epoch_key, minibatch_size)
            epoch_losses.append(loss)
        jax.block_until_ready(network)
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        
        elapsed = time.time() - t0
        
        if iteration % 10 == 0:
            avg_reward = rewards.mean()
            num_episodes = int(dones.sum())
            wins = int(jnp.sum((dones) & (infos.winner == 0)))
            losses_count = int(jnp.sum((dones) & (infos.winner == 1)))
            win_rate = wins / max(num_episodes, 1) * 100
            sps = (num_envs * num_steps) / elapsed
            print(f"Iter {iteration:4d} | Loss: {float(avg_loss):.4f} | "
                  f"Reward: {float(avg_reward):+.4f} | Episodes: {num_episodes:3d} | "
                  f"Wins: {wins:2d}/{num_episodes} ({win_rate:.0f}%) | "
                  f"SPS: {sps:7.0f} | Time: {elapsed:.2f}s")
    
    print("\nTraining complete!")
    
    # Save model
    model_path = "jax_ppo_model_env.eqx"
    eqx.tree_serialise_leaves(model_path, network)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
