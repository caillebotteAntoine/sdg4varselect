# Create by caillebotte.antoine@inrae.fr

from sdg4varselect import jnp, jit, jax


@jit
def logistic_curve_float(x, supremum: float, midpoint: float, growth_rate: float):
    return supremum / (1 + jnp.exp(-(x - midpoint) / growth_rate))


@jit
def logistic_curve(
    time: jnp.ndarray,  # shape = (J,) [None, :]
    supremum: jnp.ndarray,  # shape = (N,) [:,None]
    midpoint: jnp.ndarray,  # shape = (N,) [:,None]
    growth_rate: jnp.ndarray,  # shape = (N,) [:,None]
) -> jnp.ndarray:  # shape = (N,J)
    return supremum[:, None] / (
        1 + jnp.exp(-(time - midpoint[:, None]) / growth_rate[:, None])
    )


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2
