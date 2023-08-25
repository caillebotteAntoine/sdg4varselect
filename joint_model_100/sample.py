import numpy as np
import parametrization_cookbook.jax as pc
from sdg4varselect.data_generation import data_simulation
from model import likelihood, likelihood_array, params_weibull

from sdg4varselect import Gradient, jrd, learning_rate

plateau_jac = 300
plateau_jac_size = 0
plateau_fim = 750
total_step = plateau_fim + 300
lr = 1e-4


# ===================================================== #
# ==================== PARAMS STAR ==================== #
# ===================================================== #
def get_sample(key, params_star_weibull, N_IND, DIM_COV, cov_law="uniform"):
    # ====== DATA GENERATION ====== #
    return data_simulation(
        params=params_star_weibull,
        key=key,
        N_IND=N_IND,
        J=20,
        p=DIM_COV,
        t_min=60,
        t_max=135,
        cov_law=cov_law,
    )


def get_solver(parametrization, key, params0, data_set, N_IND):
    key1, key_out = jrd.split(key, num=2)
    # ====== SOLVER CREATION ====== #
    solver = Gradient(key1)
    solver.parametrization = parametrization
    solver.theta_reals1d = params0

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

    solver.step_size = learning_rate(
        plateau_jac, np.log(lr), plateau_jac + plateau_jac_size, 0.65, scale=0.5
    )
    solver.step_size_fisher = learning_rate(
        plateau_fim, np.log(lr), step_flat=plateau_jac + 50, scale=0.99
    )
    solver.step_size_grad = learning_rate(
        plateau_jac + 100, np.log(lr), plateau_fim + 100, 0.65, step_flat=100
    )

    # learning_rate.from_1_to_0(plateau_fim + 100, 0.65)

    # p < 200 = np.log(1e-4)
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
