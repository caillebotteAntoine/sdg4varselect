"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.outputs import MultiRunRes

from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings

from sdg4varselect.models.hd_logistic_mem import HDLogisticMixedEffectsModel

import sdg4varselect.plot as sdgplt

myModel = HDLogisticMixedEffectsModel(N=100, J=10, P=3)
p_star = myModel.new_params(
    eta1=200,
    eta2=300,
    gamma2_1=0.1,
    gamma2_2=0.1,
    mu=1200,
    sigma2=30,
    Gamma2=200,
    beta=jnp.concatenate([jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]),
)

myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

fig = sdgplt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")


algo_settings = get_GDFIM_settings(preheating=400, heating=800)


def one_estim(prngkey, model, data, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))

    algo = SPGD_FIM(prngkey_estim, 2000, algo_settings, lbd=lbd, alpha=alpha)
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"psi1": 0.01, "psi2": 0.01, "ksi": 10})

    algo.latent_variables["psi1"].adaptative_sd = True
    algo.latent_variables["psi2"].adaptative_sd = True
    algo.latent_variables["ksi"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    for var in algo.latent_variables.values():
        sdgplt.plot(var)

    return res


res = []
for i in range(1):
    res.append(one_estim(jrd.PRNGKey(i), myModel, myobs, lbd=None))
res = MultiRunRes(res)

sdgplt.plot_theta(res, myModel.DIM_LD, p_star, myModel.params_names)
sdgplt.plot_theta_hd(res, myModel.DIM_LD, p_star, myModel.params_names)
print(f"chrono = {res.chrono}")
