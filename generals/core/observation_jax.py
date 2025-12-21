"""
JAX-optimized observation using NamedTuples for better performance.

Using NamedTuples avoids Python dictionary overhead and provides:
- Better JIT compilation
- Faster attribute access
- Type safety
- Pytree compatibility for JAX transformations
"""

from typing import NamedTuple
import jax.numpy as jnp


class ObservationJax(NamedTuple):
    """
    JAX-optimized observation using NamedTuple for minimal Python overhead.
    All fields are JAX arrays for efficient vectorization and JIT compilation.
    """
    armies: jnp.ndarray
    generals: jnp.ndarray
    cities: jnp.ndarray
    mountains: jnp.ndarray
    neutral_cells: jnp.ndarray
    owned_cells: jnp.ndarray
    opponent_cells: jnp.ndarray
    fog_cells: jnp.ndarray
    structures_in_fog: jnp.ndarray
    owned_land_count: jnp.ndarray  # scalar int
    owned_army_count: jnp.ndarray  # scalar int
    opponent_land_count: jnp.ndarray  # scalar int
    opponent_army_count: jnp.ndarray  # scalar int
    timestep: jnp.ndarray  # scalar int
    priority: jnp.ndarray = jnp.int32(0)  # scalar int
    
    def as_tensor(self) -> jnp.ndarray:
        """
        Returns a 3D tensor of shape (15, rows, cols). Suitable for neural nets.
        """
        shape = self.armies.shape
        
        return jnp.stack(
            [
                self.armies,
                self.generals,
                self.cities,
                self.mountains,
                self.neutral_cells,
                self.owned_cells,
                self.opponent_cells,
                self.fog_cells,
                self.structures_in_fog,
                jnp.ones(shape, dtype=jnp.int32) * self.owned_land_count,
                jnp.ones(shape, dtype=jnp.int32) * self.owned_army_count,
                jnp.ones(shape, dtype=jnp.int32) * self.opponent_land_count,
                jnp.ones(shape, dtype=jnp.int32) * self.opponent_army_count,
                jnp.ones(shape, dtype=jnp.int32) * self.timestep,
                jnp.ones(shape, dtype=jnp.int32) * self.priority,
            ],
            axis=0,
        )
