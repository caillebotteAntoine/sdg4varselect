"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.outputs import MultiRunRes

from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings

from sdg4varselect.models.hd_logistic_mem import (
    HDLogisticMixedEffectsModel,
)

from sdg4varselect.models.hd_test import HDLogisticMixedEffectsModel

import sdg4varselect.plot as sdgplt

# myModel = HDLogisticMixedEffectsModel(N=100, J=10, P=5)
# p_star = myModel.new_params(
#     eta1=200,
#     eta2=300,
#     gamma2_1=0.1,
#     gamma2_2=0.1,
#     mu=1200,
#     sigma2=30,
#     Gamma2=200,
#     beta=jnp.concatenate([jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]),
# )

myModel = HDLogisticMixedEffectsModel(N=100, J=10, P=50)

p_star = myModel.new_params(
    mu1=0.3,
    mu2=90.0,
    mu3=7.5,
    gamma2_1=0.0025,
    gamma2_2=20,
    sigma2=0.001,
    beta=jnp.concatenate(
        [jnp.array([-3, -2, 2, 3]), jnp.zeros(shape=(myModel.P - 4,))]
    ),
)

myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

# fig = sdgplt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")


algo_settings = get_GDFIM_settings(preheating=400, heating=600)


def one_fit(theta0):
    params = myModel.parametrization.reals1d_to_params(theta0)

    algo = SPGD_FIM(jrd.PRNGKey(0), 1000, algo_settings)

    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(params.mu1),
        sd=0.001,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="phi1",
    )
    algo.latent_variables["phi1"].adaptative_sd = True
    algo.add_mcmc(
        float(params.mu2),
        sd=2,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #

    # =================== MCMC configuration ==================== #
    # algo.add_mcmc(
    #     float(params.eta1),
    #     sd=0.01,
    #     size=myModel.N,
    #     likelihood=myModel.likelihood_array,
    #     name="psi1",
    # )
    # algo.latent_variables["psi1"].adaptative_sd = True
    # algo.add_mcmc(
    #     float(params.eta2),
    #     sd=0.01,
    #     size=myModel.N,
    #     likelihood=myModel.likelihood_array,
    #     name="psi2",
    # )
    # algo.latent_variables["psi2"].adaptative_sd = True
    # algo.add_mcmc(
    #     float(params.eta2),
    #     sd=10,
    #     size=myModel.N,
    #     likelihood=myModel.likelihood_array,
    #     name="ksi",
    # )
    # algo.latent_variables["ksi"].adaptative_sd = True
    # ==================== END configuration ==================== #
    out = algo.fit(myModel, myobs, theta0, ntry=5, partial_fit=True)

    # sdgplt.plot(algo.latent_variables)
    return out


res = []
for i in range(1):
    theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
    res.append(one_fit(theta0))
res = MultiRunRes(res)

sdgplt.plot_theta(res, 6, p_star, myModel.params_names)
sdgplt.plot_theta_hd(res, 6, p_star, myModel.params_names)
print(f"chrono = {res.chrono}")
