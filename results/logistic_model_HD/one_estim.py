"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import sdgplt
from sdg4varselect.outputs import GDResults, MultiRunRes

from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings


algo_settings = get_GDFIM_settings(preheating=400, heating=600)


def one_estim(prngkey, model, data, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))

    algo = SPGD_FIM(prngkey_estim, 1000, algo_settings, lbd=lbd, alpha=alpha)
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"phi1": 0.001, "phi2": 1})

    algo.latent_variables["phi1"].adaptative_sd = True
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    # for var in algo.latent_variables.values():
    #     plt.plot(var)

    return res


if __name__ == "__main__":
    from sdg4varselect.models.hd_test import HDLogisticMixedEffectsModel

    myModel = HDLogisticMixedEffectsModel(N=100, J=10, P=50)

    p_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, _ = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

    multi_estim = MultiRunRes(
        [
            one_estim(jrd.PRNGKey(key), myModel, myobs, lbd=0.005, save_all=True)
            for key in range(2)
        ]
    )

    print(multi_estim.chrono)

    # === PLOT === #

    names = myModel.params_names

    sdgplt.plot(
        multi_estim, dim_ld=myModel.DIM_LD + 4, params_star=p_star, params_names=names
    )

    # algo.likelihood_marginal(myModel, myobs, paramslist_to_reals1d(params), size=11)
