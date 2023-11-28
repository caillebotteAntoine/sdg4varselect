import numpy as np
import parametrization_cookbook.jax as pc
from sdg4varselect.data_generation_linear import data_simulation, params_weibull

from sdg4varselect import Gradient, jrd, jnp


# ===================================================== #
# ==================== PARAMS STAR ==================== #
# ===================================================== #
def sample(
    params0_weibull,
    prng_key,
    N_IND,
    J_OBS,
    CENSORING,
):
    """return data_set, sim, prng_key"""
    # ====== DATA GENERATION ====== #
    data_set, sim, prng_key = data_simulation(
        params=params0_weibull,
        key=prng_key,
        N_IND=N_IND,
        J_OBS=J_OBS,
        t_min=0,
        t_max=1,
        censoring=CENSORING,
    )
    return data_set, sim, prng_key


def get_solver(prng_key, params0, data_set, likelihood, likelihood_array):
    key1, key_out = jrd.split(prng_key, num=2)
    N_IND, _ = data_set["Y"].shape
    (DIM_COV,) = params0["beta"].shape

    parametrization, _, prng_key = get_parametrization(prng_key, DIM_COV, "normal")
    # ====== SOLVER CREATION ====== #
    solver = Gradient(key1, parametrization, params0)

    # ============================================================ #
    # ==================== MCMC configuration ==================== #
    # ============================================================ #
    if isinstance(params0, dict):
        mu1 = params0["mu1"]
        mu2 = params0["mu2"]
    else:
        mu1 = params0.mu1
        mu2 = params0.mu2

    solver.add_mcmc(
        float(mu1),
        sd=0.5,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi1",
    )
    solver.latent_variables["phi1"].adaptative_sd = True
    solver.add_mcmc(
        float(mu2),
        sd=0.5,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi2",
    )
    solver.latent_variables["phi2"].adaptative_sd = True
    # ============================================================ #
    # ==================== END configuration ==================== #
    # ============================================================ #

    solver.add_data(**data_set)
    solver.likelihood = likelihood
    solver.add_likelihood_kwargs("time", "Y", "phi1", "phi2", "T", "delta", "cov")

    solver.add_data(parametrization=solver.parametrization)
    solver.add_likelihood_kwargs("parametrization")

    return solver, key_out


def get_parametrization(prng_key, DIM_COV, beta_type="normal", percentage=None):
    prng_key, key1, key2, key3 = jrd.split(prng_key, num=4)

    parametrization = pc.NamedTuple(
        mu1=pc.RealPositive(scale=0.1),
        mu2=pc.Real(scale=0.1),
        gamma2_1=pc.RealPositive(scale=1),
        gamma2_2=pc.RealPositive(scale=1),
        sigma2=pc.RealPositive(scale=0.01),
        alpha=pc.Real(scale=0.1),
        beta=pc.Real(scale=1, shape=(DIM_COV,)),
    )

    params_star_weibull = params_weibull(
        mu1=1,
        mu2=1.5,
        gamma2_1=2**2,
        gamma2_2=0.3**2,
        sigma2=0.1**2,
        a=110.0,
        b=35,
        alpha=0.1,
        beta=jnp.concatenate(
            [jnp.array([0.3, 0.5, 0.3, 0.5]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    params_star_weibull = params_weibull(
        mu1=1,
        mu2=1.5,
        gamma2_1=0.1**2,
        gamma2_2=0.1**2,
        sigma2=0.1**2,
        a=2.0,
        b=5,
        alpha=1.5,
        beta=jnp.concatenate(
            [jnp.array([0.3, 0.5, 0.3, 0.5]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    return parametrization, params_star_weibull, prng_key


def get_params_star(prng_key, DIM_COV, beta_type="normal", percentage=None):
    parametrization, params_star_weibull, prng_key = get_parametrization(
        prng_key, DIM_COV, beta_type, percentage
    )

    params_star_stack = np.hstack(
        [
            params_star_weibull.mu1,
            params_star_weibull.mu2,
            params_star_weibull.gamma2_1,
            params_star_weibull.gamma2_2,
            params_star_weibull.sigma2,
            params_star_weibull.alpha,
            params_star_weibull.beta,
        ]
    )

    return params_star_stack, params_star_weibull, prng_key


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    _, params_star, PRNGKey = get_parametrization(jrd.PRNGKey(0), 10)

    obs, _, PRNGKey = sample(
        params_star, prng_key=PRNGKey, N_IND=500, J_OBS=50, CENSORING=0.2
    )

    plt.plot(obs["time"].T, obs["Y"].T, "o-")

    plt.figure()
    plt.hist(obs["T"], bins=20)

    print(f'censoring = {int((1-obs["delta"].mean())*100)}%')
