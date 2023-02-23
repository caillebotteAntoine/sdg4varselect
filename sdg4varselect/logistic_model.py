import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit


"""
import parametrization_cookbook.jax as pc

parametrization = pc.NamedTuple(
    beta1=pc.RealPositive(scale=100),
    beta2=pc.Real(loc=100, scale=100),
    beta3=pc.RealPositive(scale=100),
    gamma2_1=pc.RealPositive(scale=100),
    gamma2_2=pc.RealPositive(scale=100),
)
"""


def logistic_curve_float(
    x, supremum: float, midpoint: float, growth_rate: float
) -> np.ndarray:
    return supremum / (1 + np.exp(-(x - midpoint) / growth_rate))


@jit
def logistic_curve(
    time: jnp.ndarray,  # shape = (J,) [None, :]
    supremum: jnp.ndarray,  # shape = (N,) [:,None]
    midpoint: jnp.ndarray,  # shape = (N,) [:,None]
    growth_rate: float,
) -> jnp.ndarray:  # shape = (N,J)

    return supremum[:, None] / (
        1 + jnp.exp(-(time[None, :] - midpoint[:, None]) / growth_rate)
    )


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    # Computation of the current target distrubtion score
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2
