from sample import get_solver

from sample import get_parametrization, get_sample, total_step
import numpy as np
from sdg4varselect import jnp, jrd
from model import jac_likelihood
import sdg4varselect.plot as sdgplt

DIM_COV = 50
N_IND = 90

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

    p["beta"] = jrd.uniform(jrd.PRNGKey(0), shape=p["beta"].shape, minval=-1, maxval=1)

    return p, key_new


fisher_mask = (
    jnp.arange(0, len(params0) + DIM_COV - 1) < 1000 + len(params0) - 2
)  # -2 car on veut pas de 2 paramètre beta and alpha
# fisher_mask = jnp.array([True for i in range(len(fisher_mask))])

kwargs_run_GD = {
    "prox_regul": 1.29e-3,
    "proximal_operator": True,
}


def sample(params0_weibull, prng_key):
    """return solver, data_set, key"""
    # ====== DATA GENERATION ====== #
    data_set, _, key = get_sample(prng_key, params0_weibull, N_IND, DIM_COV, cov_law)
    return data_set, key


def estim_solver(solver, verbatim=False):
    solver.verbatim = verbatim
    return solver.stochastic_gradient(
        jac_likelihood=jac_likelihood,
        fisher_mask=fisher_mask,
        smart_start=solver.step_size_fisher.step_burnin,
        p=DIM_COV,
        niter=total_step,
        **kwargs_run_GD,
    )


def estim(data_set, params0, prng_key, verbatim=False):
    solver, key = get_solver(parametrization, prng_key, params0, data_set, N_IND)

    res = estim_solver(solver, verbatim=verbatim)
    return res, solver, key


if __name__ == "__main__":
    cov_law = "uniform"
    data_set, prng_key = sample(params_star_weibull, jrd.PRNGKey(0))
    res, solver, _ = estim(data_set, params0, prng_key, verbatim=True)

    sdgplt.plot_params(
        x=res.theta,
        x_star=np.array(params_star_stack),
        p=DIM_COV,
        names=solver.params_names,
        logscale=False,
    )

    sdgplt.plot_params_hd(res.theta, p=DIM_COV, location="right")
