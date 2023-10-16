from sample import get_solver

from sample import get_parametrization, get_sample
import numpy as np
from sdg4varselect import jnp, jrd, learning_rate
from model import jac_likelihood
import sdg4varselect.plot as sdgplt

DIM_COV = 200
N_IND = 100
J_OBS = 20

cov_law = "uniform"

beta_type = {"clever_uniform": "clever", "NIRS": "NIRS", "uniform": "normal"}

parametrization, params_star_weibull = get_parametrization(DIM_COV, beta_type[cov_law])

params_star_stack = jnp.hstack(
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

params0 = {
    "mu1": 0.5,  # 1
    "mu2": 50.0,  # 2
    "mu3": 3.0,  # 3
    "gamma2_1": 0.00025,  # 4
    "gamma2_2": 2.0,  # 5
    "sigma2": 0.0001,  # 6
    "alpha": 5.0,  # 7
    "beta": np.random.uniform(-1, 1, size=DIM_COV),
}


def get_random_params0(prng_key, error=0.2):
    p = params0.copy()
    for key in p:
        key_new, prng_key = jrd.split(prng_key, 2)
        p[key] *= float(jrd.uniform(key_new, minval=1.0 - error))

    key_new, prng_key = jrd.split(prng_key, 2)
    p["beta"] = jrd.uniform(prng_key, shape=p["beta"].shape, minval=-1, maxval=1)

    return p, key_new


fisher_mask = (
    jnp.arange(0, len(params0) + DIM_COV - 1) < len(params0) - 2
)  # -2 car on veut pas de 2 paramètre beta and alpha
# fisher_mask = jnp.array([True for i in range(len(fisher_mask))])


def sample(params0_weibull, prng_key):
    """return solver, data_set, key"""
    # ====== DATA GENERATION ====== #
    data_set, _, key = get_sample(
        prng_key, params0_weibull, N_IND, DIM_COV, J_OBS, cov_law
    )
    return data_set, key


def set_solver_run_parameters(
    solver,
    activate_fim=False,
    activate_jac_approx=True,
    lr=1e-8,
    # Grad
    plateau_grad=400,
    scale_grad=1,
    plateau_grad_size=100,
    # Jac
    plateau_jac=300,
    plateau_jac_size=50,
    scale_jac=0.5,
    # Fim
    plateau_fim=750,
    plateau_fim_size=50,
    scale_fim=0.95,
):
    if activate_jac_approx:
        solver.step_size = learning_rate(
            plateau_jac,
            np.log(lr),
            plateau_jac + plateau_jac_size,
            0.65,
            scale=scale_jac,
        )
    else:
        solver.step_size = learning_rate.one()

    if activate_fim:
        solver.step_size_fisher = learning_rate(
            plateau_fim,
            np.log(lr),
            plateau_fim + plateau_fim_size,
            0.65,
            # step_flat=plateau_jac + 100,
            scale=scale_fim,
        )
    else:
        solver.step_size_fisher = learning_rate.zero()

    solver.step_size_grad = learning_rate(
        plateau_grad,
        np.log(lr),
        plateau_grad + plateau_grad_size,
        0.65,
        scale=scale_grad,
        # step_flat=100,
    )

    # solver.step_size = solver.step_size_fisher
    # solver.step_size_grad = solver.step_size_fisher
    # learning_rate.from_1_to_0(plateau_fim + 100, 0.65)
    return solver


def estim_solver(solver, niter, kwargs_run_GD, verbatim=False, **run_parameters):
    solver.verbatim = verbatim

    solver = set_solver_run_parameters(solver, **run_parameters)

    res = solver.stochastic_gradient(
        jac_likelihood=jac_likelihood,
        fisher_mask=fisher_mask,
        smart_start=solver.step_size_fisher.step_burnin,
        p=DIM_COV,
        niter=niter,
        **kwargs_run_GD,
    )

    return res, solver


def estim(
    data_set, params0, prng_key, niter, kwargs_run_GD, verbatim=False, **run_parameters
):
    solver, key = get_solver(parametrization, prng_key, params0, data_set, N_IND)

    res, solver = estim_solver(
        solver, niter, kwargs_run_GD=kwargs_run_GD, verbatim=verbatim, **run_parameters
    )
    return res, solver, key


if __name__ == "__main__":
    pass
