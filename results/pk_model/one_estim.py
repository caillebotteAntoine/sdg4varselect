import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import SPG_FIM
from sdg4varselect.exceptions import sdg4vsNanError

algo_settings = SPG_FIM.settings(
    step_size_grad={
        "learning_rate": 1e-8,
        "preheating": 1000,
        "heating": 1400,
        "max": 0.9,
    },
    step_size_approx_sto={
        "learning_rate": 1e-8,
        "preheating": 1000,
        "heating": 2000,
        "max": 1,
    },
    step_size_fisher={
        "learning_rate": 1e-8,
        "preheating": 1000,
        "heating": 2000,
        "max": 0.9,
    },
)


def estim(PRNGKey, model, dh, theta0, algo_settings, lbd=None, alpha=1.0):
    # _, _ = sdgplt.plot_sample(data_set, sim, params_star, 2000, 80, 35)

    params0 = model.parametrization.reals1d_to_params(theta0)
    algo = SPG_FIM(PRNGKey, dh, algo_settings, lbd=lbd, alpha=alpha)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(params0.mu1),
        sd=0.1,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi1",
    )
    algo.latent_variables["phi1"].adaptative_sd = True
    algo.add_mcmc(
        float(params0.mu2),
        sd=0.1,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(
        model.jac_likelihood,
        niter=2000,
        DIM_HD=model.DIM_HD,
        theta0_reals1d=theta0,
        # partial_fit=True,
    )

    return res, algo


def one_estim(PRNGKey, model, dh, algo_settings, lbd=None, alpha=1.0):
    PRNGKey_theta, PRNGKey_estim, PRNGKey_likelihoohd = jrd.split(PRNGKey, 3)
    theta0 = 0.2 * jrd.normal(PRNGKey_theta, shape=(model.parametrization.size,))

    try:
        res_estim, algo = estim(
            PRNGKey_estim, model, dh, theta0, algo_settings, lbd=lbd, alpha=alpha
        )
    except sdg4vsNanError as err:
        print(err)
        return sdg4vsNanError

    res = algo.labelswitch(res_estim)
    return algo.estim_res(
        theta=jnp.array([model.reals1d_to_hstack_params(t) for t in res.theta]),
        FIM=res.FIM,
        grad=res.grad,
        likelihood=algo.likelihood_marginal(model, PRNGKey_likelihoohd, res.theta[-1]),
    )


if __name__ == "__main__":
    from sdg4varselect.models.pharmacokinetic import (
        pharma_JM,
        sample_one,
        get_params_star,
    )
    from time import time

    model = pharma_JM(N=100, J=5, DIM_HD=5)

    dh = sample_one(jrd.PRNGKey(int(time())), model, weibull_censoring_loc=2000)

    multi_estim = [
        one_estim(jrd.PRNGKey(key), model, dh, algo_settings, lbd=None)
        for key in range(20)
    ]
    while sdg4vsNanError in multi_estim:
        multi_estim.remove(sdg4vsNanError)

    # === PLOT === #
    from sdg4varselect.plot import (
        plot_theta,
        plot_theta_HD,
    )

    params_star = get_params_star(model.DIM_HD)

    plot_theta(multi_estim, model.DIM_LD, params_star, model.params_names)
    plot_theta_HD(multi_estim, model.DIM_LD, params_star, model.params_names)

    # sdgplt.plot_mcmc(algo.mcmc)
