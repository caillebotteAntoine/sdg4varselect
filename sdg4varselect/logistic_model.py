import jax.numpy as jnp
from jax import jit
from math import pi


@jit
def logistic_curve(
    x, supremum: float, midpoint: float, growth_rate: float
) -> jnp.ndarray:
    return supremum / (1 + jnp.exp(-(x - midpoint) / growth_rate))


@jit
def partial_loglikelihood(i: int, theta, Y, time, phi1, phi2, phi3) -> jnp.ndarray:
    pred = logistic_curve(time, phi1[i], phi2[i], phi3[i])
    out = jnp.sum(pow(Y[i] - pred, 2))
    return jnp.sum(-out / (2 * theta.sigma2))


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    # Computation of the current target distrubtion score
    out = jnp.log(2 * pi * variance) + jnp.power(data - mean, 2) / variance
    return -jnp.sum(out) / 2


@jit
def model(time, phi1, phi2, phi3, **kwargs) -> jnp.ndarray:
    N = len(phi1)
    out = [logistic_curve(time, phi1[i], phi2[i], phi3[i]) for i in range(N)]
    return jnp.array(out)


@jit
def loss(theta, Y, time, phi1, phi2, phi3):
    latent_prior = (
        gaussian_prior(phi1, theta.beta1, theta.gamma2_1)
        + gaussian_prior(phi2, theta.beta2, theta.gamma2_2)
        + gaussian_prior(phi3, theta.beta3, 1)
    )

    pred = model(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2))
    out = latent_prior - out / (2 * theta.sigma2[0])

    return jnp.sum(out) / len(phi1)
