"""
Neural network architecture for Generals.io PPO agent.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx


class PolicyValueNetwork(eqx.Module):
    """
    Convolutional policy-value network for 4x4 Generals grid.
    
    Architecture:
    - 4 convolutional layers for feature extraction
    - Separate policy head (outputs action logits)
    - Separate value head (outputs state value)
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    policy_conv: eqx.nn.Conv2d
    value_conv: eqx.nn.Conv2d
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear

    def __init__(self, key, grid_size=4, channels=(32, 32, 32, 16)):
        """
        Initialize the network.
        
        Args:
            key: JAX random key
            grid_size: Size of the game grid
            channels: Number of channels in each conv layer
        """
        keys = jrandom.split(key, 6)

        self.conv1 = eqx.nn.Conv2d(9, channels[0], kernel_size=3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, key=keys[2])
        self.conv4 = eqx.nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1, key=keys[3])
        
        # Policy head: 9 channels = 4 dirs + 4 half-moves + 1 pass
        self.policy_conv = eqx.nn.Conv2d(channels[3], 9, kernel_size=1, key=keys[4])
        
        # Value head
        self.value_conv = eqx.nn.Conv2d(channels[3], 4, kernel_size=1, key=keys[5])
        self.value_linear1 = eqx.nn.Linear(grid_size * grid_size * 4, 64, key=keys[4])
        self.value_linear2 = eqx.nn.Linear(64, 1, key=keys[5])

    def __call__(self, obs, mask, key, action=None):
        """
        Forward pass through the network.
        
        Args:
            obs: Observation array [9, H, W]
            mask: Valid action mask [H, W, 4]
            key: Random key for action sampling
            action: If provided, evaluate this action. Otherwise sample.
            
        Returns:
            (action, value, log_prob, entropy)
        """
        grid_size = obs.shape[-1]

        # Feature extraction backbone
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
        logits = self.policy_conv(x)  # [9, H, W]
        
        # Apply action mask
        mask_t = jnp.transpose(mask, (2, 0, 1))  # [4, H, W]
        mask_penalty = (1 - mask_t) * -1e9
        # Add pass action (always valid)
        combined_mask = jnp.concatenate([
            mask_penalty,  # 4 directions
            mask_penalty,  # 4 half-move directions
            jnp.zeros((1, grid_size, grid_size))  # pass action
        ], axis=0)
        logits = (logits + combined_mask).reshape(-1)

        grid_cells = grid_size * grid_size

        if action is None:
            # Sample action
            idx = jrandom.categorical(key, logits)
            direction, position = idx // grid_cells, idx % grid_cells
            row, col = position // grid_size, position % grid_size
            is_pass = direction == 8
            is_half = (direction >= 4) & (direction < 8)
            actual_dir = jnp.where(is_pass, 0, jnp.where(is_half, direction - 4, direction))
            action = jnp.array([is_pass, row, col, actual_dir, is_half], dtype=jnp.int32)
        else:
            # Compute index from provided action
            is_pass, row, col, direction, is_half = action
            encoded_dir = jnp.where(is_pass > 0, 8, jnp.where(is_half > 0, direction + 4, direction))
            idx = encoded_dir * grid_cells + row * grid_size + col

        # Compute log probability and entropy
        log_probs = jax.nn.log_softmax(logits)
        logprob = log_probs[idx]
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs)

        return action, value, logprob, entropy


def obs_to_array(obs):
    """
    Convert Observation namedtuple to network input array.
    
    Args:
        obs: Observation from environment
        
    Returns:
        Array [9, H, W] with stacked observation channels
    """
    return jnp.stack([
        obs.armies,
        obs.generals,
        obs.cities,
        obs.mountains,
        obs.neutral_cells,
        obs.owned_cells,
        obs.opponent_cells,
        obs.fog_cells,
        obs.structures_in_fog
    ], axis=0).astype(jnp.float32)

