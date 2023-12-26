# Create by caillebotte.antoine@inrae.fr

from sdg4varselect import jnp, jit, jax


@jit
def linear_curve_float(x, intercept: float, slope: float):
    return intercept + slope * x


@jit
def linear_curve(
    time: jnp.ndarray,  # shape = (N,J)
    intercept: jnp.ndarray,  # shape = (N,) [:,None]
    slope: jnp.ndarray,  # shape = (N,) [:,None]
) -> jnp.ndarray:  # shape = (N,J)
    return intercept[:, None] + time * slope[:, None]


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


if __name__ == "__main__":
    from sdg4varselect import jrd
    from matplotlib import pyplot as plt

    N = 10
    J = 5

    time = jnp.arange(J) * 365
    time = jnp.concatenate(jnp.array([[time]] * N), axis=0)
    time += jrd.choice(jrd.PRNGKey(0), jnp.arange(1, 365), time.shape)
    time /= time.shape[1] * 365

    print(time)

    intercept = jrd.uniform(jrd.PRNGKey(0), (N,))
    slope = 15 * jrd.uniform(jrd.PRNGKey(0), (N,))

    Y = linear_curve(time, intercept, slope) + 0.5 * jrd.normal(
        jrd.PRNGKey(0), time.shape
    )

    plt.plot(time.T, Y.T, "o-")
