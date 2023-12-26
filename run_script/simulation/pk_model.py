# Create by caillebotte.antoine@inrae.fr

from sdg4varselect import jnp, jit, jax


@jit
def pk_curve_float(t, D: float, ka: float, Cl: float, V: float):
    return D * ka / (V * ka - Cl) * (jnp.exp(-ka * t) - jnp.exp(-Cl / V * t))


@jit
def pk_curve(
    time: jnp.ndarray,  # shape = (N,J)
    D: jnp.ndarray,  # shape = (1,) [:,None]
    ka: jnp.ndarray,  # shape = (N,) [:,None]
    Cl: jnp.ndarray,  # shape = (N,) [:,None]
    V: jnp.ndarray,  # shape = (N,) [:,None]
) -> jnp.ndarray:  # shape = (N,J)
    return (
        D
        * ka[:, None]
        / (V[:, None] * (ka[:, None] - Cl[:, None] / V[:, None]))
        * (jnp.exp(-ka[:, None] * time) - jnp.exp(-Cl[:, None] / V[:, None] * time))
    )


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


if __name__ == "__main__":
    from sdg4varselect import jrd
    from matplotlib import pyplot as plt
    from collections import namedtuple
    from data_generation import mem_simulation

    params = namedtuple(
        "params",
        ("mu1", "mu2", "V", "gamma2_1", "gamma2_2", "sigma2"),
    )
    parameters = params(8, 6, 40, 0.2, 0.1, 1e-3)

    time = jnp.array([0.05, 0.15, 0.25, 0.4, 0.5, 0.8, 1, 2, 7, 12, 24, 40])

    random_effects = {"ka": ("mu1", "gamma2_1"), "Cl": ("mu2", "gamma2_2")}
    fixed_effets = {"V": "V"}
    obs = {"time": time}

    obs, sim, PRNGKey = mem_simulation(
        parameters,
        jrd.PRNGKey(0),
        10,
        "sigma2",
        pk_curve,
        random_effects,
        fixed_effets,
        fct_kwargs={"time": obs["time"], "D": -100},
    )

    plt.plot(time.T, obs["Y"].T, "o-")
