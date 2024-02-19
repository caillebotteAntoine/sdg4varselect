"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd

# import jax.numpy as jnp

from sdg4varselect import SPG_FIM
from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.outputs import SDGResults

algo_settings = SPG_FIM.SPGfimSettings(
    step_size_grad={
        "learning_rate": 1e-12,
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
    max_iter=2000,
)


def estim(prngkey, model, dh, theta0, lbd=None, alpha=1.0):
    params0 = model.parametrization.reals1d_to_params(theta0)
    algo = SPG_FIM(prngkey, dh, algo_settings, lbd=lbd, alpha=alpha)
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
        DIM_HD=model.DIM_HD,
        theta0_reals1d=theta0,
        ntry=5,
        partial_fit=False,
    )

    return res, algo


def one_estim(prngkey, model, dh, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim, prngkey_likelihood = jrd.split(prngkey, 3)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    try:
        res_estim, algo = estim(prngkey_estim, model, dh, theta0, lbd=lbd, alpha=alpha)
    except sdg4vsNanError as err:
        print(err)
        return sdg4vsNanError

    res = SDGResults.compute_with_model(prngkey_likelihood, algo, model, res_estim)
    return res if save_all else SDGResults.make_it_lighter(res)


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.pharmacokinetic import (
        PharmaJM,
        get_params_star,
    )

    myModel = PharmaJM(N=100, J=10, DIM_HD=10)
    params_star = get_params_star(myModel.DIM_HD)

    myDH = sample_model(
        jrd.PRNGKey(0), params_star, myModel, weibull_censoring_loc=2000
    )

    multi_estim = [
        one_estim(jrd.PRNGKey(key), myModel, myDH, lbd=None, save_all=True)
        for key in range(10)
    ]
    while sdg4vsNanError in multi_estim:
        multi_estim.remove(sdg4vsNanError)

    # === PLOT === #

    # prngkey_theta, prngkey_estim, prngkey_likelihood = jrd.split(jrd.PRNGKey(3), 3)
    # theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(myModel.parametrization.size,))

    # res, algo = estim(prngkey_estim, myModel, myDH, theta0, lbd=None)
    # multi_estim = SDGResults.compute_with_model(prngkey_likelihood, algo, myModel, res)

    # for var in algo.latent_variables.values():
    #     sdgplt.plot_mcmc(var)

    params_star = get_params_star(myModel.DIM_HD)

    sdgplt.plot_theta(multi_estim, myModel.DIM_LD, params_star, myModel.params_names)
    sdgplt.plot_theta_HD(multi_estim, myModel.DIM_LD, params_star, myModel.params_names)

    # sdgplt.plot_mcmc(algo.mcmc)
