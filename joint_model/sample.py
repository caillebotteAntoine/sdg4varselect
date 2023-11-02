import numpy as np
import parametrization_cookbook.jax as pc
from sdg4varselect.data_generation import data_simulation
from model import likelihood, likelihood_array, params_weibull

from sdg4varselect import Gradient, jrd


# ===================================================== #
# ==================== PARAMS STAR ==================== #
# ===================================================== #
def sample(
    params0_weibull,
    prng_key,
    N_IND,
    DIM_COV,
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
        p=DIM_COV,
        t_min=60,
        t_max=135,
        cov_law="uniform",
        censoring=CENSORING,
    )
    return data_set, sim, prng_key


def get_solver(prng_key, params0, data_set):
    key1, key_out = jrd.split(prng_key, num=2)
    N_IND, _ = data_set["Y"].shape
    (DIM_COV,) = params0["beta"].shape

    parametrization, _ = get_parametrization(DIM_COV, "normal")
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
        sd=0.001,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi1",
    )
    solver.latent_variables["phi1"].adaptative_sd = True
    solver.add_mcmc(
        float(mu2),
        sd=5,
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
    solver.add_likelihood_kwargs("time", "Y", "phi1", "phi2", "T", "cov")

    solver.add_data(parametrization=solver.parametrization)
    solver.add_likelihood_kwargs("parametrization")

    return solver, key_out


def get_parametrization(DIM_COV, beta_type="normal"):
    beta = np.zeros(shape=(DIM_COV,))

    if beta_type == "normal":
        beta[0] = -2
        if DIM_COV > 1:
            beta[1] = -3
        if DIM_COV > 2:
            beta[2] = 3
        if DIM_COV > 3:
            beta[3] = 2

    elif beta_type == "clever":
        beta[0] = -4
        beta[1] = -2
        beta[2] = 4
        beta[3] = 2

    elif beta_type == "NIRS":
        beta[2] = -2
        beta[30] = -10
        beta[53] = 4
        beta[77] = 2

    parametrization = pc.NamedTuple(
        mu1=pc.RealPositive(scale=0.5),
        mu2=pc.Real(scale=100),
        mu3=pc.RealPositive(scale=10),
        gamma2_1=pc.RealPositive(scale=0.001),
        gamma2_2=pc.RealPositive(scale=10),
        sigma2=pc.RealPositive(scale=0.001),
        alpha=pc.Real(scale=10),
        beta=pc.Real(scale=1, shape=(DIM_COV,)),
    )

    params_star_weibull = params_weibull(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        a=80.0,
        b=35,
        alpha=11.11,
        beta=beta,
    )

    return parametrization, params_star_weibull


def get_params_star(DIM_COV, beta_type="normal"):
    parametrization, params_star_weibull = get_parametrization(DIM_COV, "normal")

    params_star_stack = np.hstack(
        [
            params_star_weibull.mu1,
            params_star_weibull.mu2,
            params_star_weibull.mu3,
            params_star_weibull.gamma2_1,
            params_star_weibull.gamma2_2,
            params_star_weibull.sigma2,
            params_star_weibull.alpha,
            params_star_weibull.beta,
        ]
    )

    return params_star_stack, params_star_weibull


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt

    # from one_run import DIM_COV, N_IND, J_OBS
    _, params_star_weibull = get_parametrization(20, "normal")

    dt, sim, PRNGKey = sample(
        params_star_weibull,
        prng_key=jrd.PRNGKey(0),
        N_IND=500,
        DIM_COV=5,
        J_OBS=50,
        CENSORING=0.2,
    )

    print(dt["delta"].mean())

    sdgplt.plt.plot(dt["time"], dt["Y"].T)

    sdgplt.figure()
    sdgplt.plt.hist(dt["T"], bins=20)

    sdgplt.figure()
    sdgplt.plt.hist(sim["T uncensored"], bins=20)

    # sdgplt.figure()
    # sdgplt.plt.hist(sim["phi2"])
