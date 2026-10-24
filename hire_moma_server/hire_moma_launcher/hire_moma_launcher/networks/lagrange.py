from functools import partial
from typing import Optional, Sequence

import chex
import flax.linen as nn
import jax.numpy as jnp


class LagrangeMultiplier(nn.Module):
    """Learnable multiplier, used as the SAC temperature (alpha)."""

    init_value: float = 1.0
    constraint_shape: Sequence[int] = ()
    constraint_type: str = "eq"  # "eq", "leq" or "geq"
    parameterization: Optional[str] = None  # "softplus", "exp" or None (equality only)

    @nn.compact
    def __call__(self, *, lhs: Optional[jnp.ndarray] = None, rhs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        init_value = self.init_value
        if self.constraint_type != "eq":
            assert init_value > 0, "Inequality constraints need a positive initial multiplier."
            if self.parameterization == "softplus":
                init_value = jnp.log(jnp.exp(init_value) - 1)
            elif self.parameterization == "exp":
                init_value = jnp.log(init_value)
            elif self.parameterization != "none":
                raise ValueError(f"Invalid multiplier parameterization {self.parameterization}")
        else:
            assert self.parameterization is None, "Equality constraints have no parameterization."

        multiplier = self.param("lagrange", lambda _, shape: jnp.full(shape, init_value), self.constraint_shape)
        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = nn.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = jnp.exp(multiplier)

        if lhs is None:
            return multiplier

        if rhs is None:
            rhs = jnp.zeros_like(lhs)
        diff = lhs - rhs
        chex.assert_equal_shape([diff, multiplier])
        if self.constraint_type == "leq":
            return -multiplier * diff
        return multiplier * diff


GeqLagrangeMultiplier = partial(LagrangeMultiplier, constraint_type="geq", parameterization="softplus")
