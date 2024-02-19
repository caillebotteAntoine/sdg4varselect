"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd

# import jax.numpy as jnp

import sdg4varselect.plot as plt
from sdg4varselect.outputs import GDResults, MultiRunRes

from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings


algo_settings = get_GDFIM_settings(preheating=1000, heating=1400)


def one_estim(prngkey, model, data, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))
    params0 = model.parametrization.reals1d_to_params(theta0)

    algo = SPGD_FIM(prngkey_estim, 2000, algo_settings, lbd=lbd, alpha=alpha)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(params0.mu1),
        sd=0.001,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi1",
    )
    algo.latent_variables["phi1"].adaptative_sd = True
    algo.add_mcmc(
        float(params0.mu2),
        sd=2,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=True)

    # for var in algo.latent_variables.values():
    #     plt.plot(var)

    return res if save_all else GDResults.make_it_lighter(res)


if __name__ == "__main__":
    from sdg4varselect.models.wcox_mem_joint_model import (
        create_logistic_weibull_jm,
        get_params_star,
    )

    myModel = create_logistic_weibull_jm(100, 5, 10)
    p_star = get_params_star(myModel)

    myobs, _ = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

    multi_estim = MultiRunRes(
        [
            one_estim(jrd.PRNGKey(key), myModel, myobs, lbd=0.1, save_all=True)
            for key in range(2)
        ]
    )

    print(multi_estim.chrono)

    # === PLOT === #

    names = myModel.params_names

    plt.plot(multi_estim, dim_ld=7, params_star=p_star, params_names=names)

    # sdgplt.plot_mcmc(algo.mcmc)
