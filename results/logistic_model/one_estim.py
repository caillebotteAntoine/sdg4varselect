"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

# import jax.numpy as jnp

from sdg4varselect import sdgplt
from sdg4varselect.outputs import MultiRunRes

from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings


algo_settings = get_GDFIM_settings(preheating=1000, heating=1400)


def one_estim(prngkey, model, data, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    algo = SPGD_FIM(prngkey_estim, 2000, algo_settings, lbd=lbd, alpha=alpha)
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"phi1": 0.001, "phi2": 2})

    algo.latent_variables["phi1"].adaptative_sd = True
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    # for var in algo.latent_variables.values():
    #     plt.plot(var)

    return res


if __name__ == "__main__":
    from sdg4varselect.models import create_cox_mem_jm, logisticMEM

    myModel = create_cox_mem_jm(logisticMEM, 100, 5, 10)

    p_star = myModel.new_params(
        mean_latent={"mu1": 0.3, "mu2": 90.0},
        mu3=7.5,
        cov_latent=jnp.diag(jnp.array([0.0025, 20])),
        var_residual=0.001,
        alpha=110.1,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, _ = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=7700)

    multi_estim = MultiRunRes(
        [
            one_estim(jrd.PRNGKey(key), myModel, myobs, lbd=None, save_all=True)
            for key in range(20)
        ]
    )

    print(multi_estim.chrono)

    # === PLOT === #
    sdgplt.plot(
        multi_estim,
        dim_ld=myModel.DIM_LD,
        params_star=myModel.hstack_params(p_star),
        # params_names=myModel.params_names,
    )

    # sdgplt.plot_mcmc(algo.mcmc)
