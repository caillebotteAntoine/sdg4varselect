from sample import get_solver

from sample import get_parametrization, get_sample_and_solver
import numpy as np
from sdg4varselect import jnp, jrd
from model import jac_likelihood
import sdg4varselect.plot as sdgplt

DIM_COV = 500
N_IND = 100

parametrization, params_star_weibull = get_parametrization(DIM_COV)

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

fisher_mask = (
    jnp.arange(0, len(params0) + DIM_COV - 1) < len(params0) - 2
)  # -2 car on veut pas de 2 paramètre beta and alpha
# fisher_mask = jnp.array([True for i in range(len(fisher_mask))])

kwargs_run_GD = {
    "prox_regul": 1.29e-3,
    "proximal_operator": True,
}


def estim(solver, verbatim=False):
    solver.verbatim = verbatim
    return solver.stochastic_gradient(
        jac_likelihood=jac_likelihood,
        fisher_preconditionner=True,
        fisher_mask=fisher_mask,
        smart_start=1500,
        p=DIM_COV,
        niter=2500,
        **kwargs_run_GD,
    )


def sample_and_estim(params0, prng_key, verbatim=False):
    solver, _, key = get_sample_and_solver(
        parametrization,
        prng_key,
        params0,
        params_star_weibull,
        N_IND,
        DIM_COV,
        plateau_start=1500,
        plateau_stop=1500 + 100,
        step_size=np.log(1e-4),
    )

    res = estim(solver, verbatim=verbatim)

    return res, solver, key


if __name__ == "__main__":
    res, solver, _ = sample_and_estim(params0, jrd.PRNGKey(0), verbatim=True)

    sdgplt.plot_params(
        x=res.theta,
        x_star=np.array(params_star_stack),
        p=DIM_COV,
        names=solver.params_names,
        logscale=False,
    )

    sdgplt.plot_params_hd(res.theta, p=DIM_COV, location="right")
