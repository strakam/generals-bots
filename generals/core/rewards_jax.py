import jax
import jax.numpy as jnp

from generals.core.observation_jax import ObservationJax


def compute_num_cities_owned(observation: ObservationJax) -> jnp.ndarray:
    """Count number of cities owned by the agent."""
    owned_cities_mask = observation.cities & observation.owned_cells
    num_cities_owned = jnp.sum(owned_cities_mask)
    return num_cities_owned.astype(jnp.float32)


def compute_num_generals_owned(observation: ObservationJax) -> jnp.ndarray:
    """Count number of generals owned by the agent."""
    owned_generals_mask = observation.generals & observation.owned_cells
    num_generals_owned = jnp.sum(owned_generals_mask)
    return num_generals_owned.astype(jnp.float32)


@jax.jit
def calculate_army_size(castles: jnp.ndarray, ownership: jnp.ndarray) -> jnp.ndarray:
    """Calculate total army size in castles (cities/generals) owned by the player."""
    return jnp.sum(castles * ownership).astype(jnp.float32)


@jax.jit
def city_reward_fn(
    prior_obs: ObservationJax, 
    prior_action: jnp.ndarray, 
    obs: ObservationJax,
    shaping_weight: float = 0.3
) -> jnp.ndarray:
    """
    Reward function that shapes the reward based on the number of cities owned.
    
    Args:
        shaping_weight: Weight for city change shaping term
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # If game is done, don't shape the reward
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    
    city_now = calculate_army_size(obs.cities, obs.owned_cells)
    city_prev = calculate_army_size(prior_obs.cities, prior_obs.owned_cells)
    city_change = city_now - city_prev
    
    shaped_reward = original_reward + shaping_weight * city_change
    
    return jnp.where(game_done, original_reward, shaped_reward)


@jax.jit
def ratio_reward_fn(
    prior_obs: ObservationJax, 
    prior_action: jnp.ndarray, 
    obs: ObservationJax,
    clip_value: float = 1.5,
    shaping_weight: float = 0.5
) -> jnp.ndarray:
    """
    Reward function that shapes based on army ratio between player and opponent.
    
    Args:
        clip_value: Maximum ratio for clipping
        shaping_weight: Weight for ratio shaping term
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # If game is done, don't shape the reward
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    
    def calculate_ratio_reward(my_army: jnp.ndarray, opponent_army: jnp.ndarray) -> jnp.ndarray:
        ratio = my_army / jnp.maximum(opponent_army, 1.0)  # Avoid division by zero
        ratio = jnp.log(ratio) / jnp.log(clip_value)
        return jnp.clip(ratio, -1.0, 1.0)
    
    prev_ratio_reward = calculate_ratio_reward(
        prior_obs.owned_army_count.astype(jnp.float32), 
        prior_obs.opponent_army_count.astype(jnp.float32)
    )
    current_ratio_reward = calculate_ratio_reward(
        obs.owned_army_count.astype(jnp.float32), 
        obs.opponent_army_count.astype(jnp.float32)
    )
    ratio_reward = current_ratio_reward - prev_ratio_reward
    
    shaped_reward = original_reward + shaping_weight * ratio_reward
    
    return jnp.where(game_done, original_reward, shaped_reward)


@jax.jit
def win_lose_reward_fn(
    prior_obs: ObservationJax, 
    prior_action: jnp.ndarray, 
    obs: ObservationJax
) -> jnp.ndarray:
    """
    Simple reward function based on generals owned with small bonus for splitting.
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # Encourage splitting a bit
    split_bonus = jnp.where(prior_action[4] == 1, 0.0015, 0.0)
    
    return original_reward + split_bonus


@jax.jit
def composite_reward_fn(
    prior_obs: ObservationJax, 
    prior_action: jnp.ndarray, 
    obs: ObservationJax,
    city_weight: float = 0.4,
    ratio_weight: float = 0.3,
    maximum_army_ratio: float = 1.6,
    maximum_land_ratio: float = 1.3
) -> jnp.ndarray:
    """
    Composite reward function combining multiple reward signals.
    
    Combines:
    - Base win/lose reward (generals owned)
    - Army ratio reward
    - Land ratio reward  
    - City capture reward
    - Split action bonus
    
    Args:
        city_weight: Weight for city reward
        ratio_weight: Weight for ratio rewards (army and land)
        maximum_army_ratio: Maximum army ratio for clipping
        maximum_land_ratio: Maximum land ratio for clipping
    """
    original_reward = (
        compute_num_generals_owned(obs) - compute_num_generals_owned(prior_obs)
    )
    
    # If game is done, don't shape the reward (except split bonus)
    game_done = (obs.owned_army_count == 0) | (obs.opponent_army_count == 0)
    
    def calculate_ratio_reward(
        mine: jnp.ndarray, 
        opponents: jnp.ndarray, 
        max_ratio: float
    ) -> jnp.ndarray:
        ratio = mine / jnp.maximum(opponents, 1.0)  # Avoid division by zero
        ratio = jnp.log(ratio) / jnp.log(max_ratio)
        return jnp.clip(ratio, -1.0, 1.0)
    
    # Army ratio reward
    previous_army_ratio = calculate_ratio_reward(
        prior_obs.owned_army_count.astype(jnp.float32),
        prior_obs.opponent_army_count.astype(jnp.float32),
        maximum_army_ratio
    )
    current_army_ratio = calculate_ratio_reward(
        obs.owned_army_count.astype(jnp.float32),
        obs.opponent_army_count.astype(jnp.float32),
        maximum_army_ratio
    )
    army_reward = current_army_ratio - previous_army_ratio
    
    # Land ratio reward
    previous_land_ratio = calculate_ratio_reward(
        prior_obs.owned_land_count.astype(jnp.float32),
        prior_obs.opponent_land_count.astype(jnp.float32),
        maximum_land_ratio
    )
    current_land_ratio = calculate_ratio_reward(
        obs.owned_land_count.astype(jnp.float32),
        obs.opponent_land_count.astype(jnp.float32),
        maximum_land_ratio
    )
    land_reward = current_land_ratio - previous_land_ratio
    
    # City reward
    city_reward = compute_num_cities_owned(obs) - compute_num_cities_owned(prior_obs)
    
    # Split bonus
    split_bonus = jnp.where(prior_action[4] == 1, 0.003, 0.0)
    
    # Combine all rewards
    shaped_reward = (
        original_reward
        + ratio_weight * army_reward
        + city_weight * city_reward
        + ratio_weight * land_reward
        + split_bonus
    )
    
    # If game done, only return original reward + split bonus
    return jnp.where(game_done, original_reward + split_bonus, shaped_reward)
