"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

# import jax.numpy as jnp

import matplotlib.pyplot as plt
from sdg4varselect import sdgplt
from sdg4varselect.outputs import GDResults, MultiRunRes

from sdg4varselect.algo import SGD_FIM, get_GDFIM_settings, GradFimSettings


from sdg4varselect.models.linear_me_model import LinearLatentModel


myModel = LinearLatentModel(N=100, J=5)


p_star = myModel.new_params(
    mu1=3,
    gamma2_1=1,
    intercept=1.5,
    sigma2=1,
)


myobs, _ = myModel.sample(p_star, jrd.PRNGKey(0))

plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")

algo_settings = get_GDFIM_settings(preheating=400, heating=600)

learning_rate = 1e-8
preheating = 400
heating = 1000

algo_settings = GradFimSettings(
    step_size_grad={
        "learning_rate": learning_rate,
        "preheating": preheating,
        "heating": heating,
        "max": 0.9,
    },
    step_size_approx_sto={
        "learning_rate": learning_rate,
        "preheating": preheating,
        "heating": None,
        "max": 1,
    },
    step_size_fisher={
        "learning_rate": learning_rate,
        "preheating": preheating,
        "heating": None,
        "max": 0.9,
    },
)


def one_estim(prngkey, model, data, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))
    params0 = model.parametrization.reals1d_to_params(theta0)

    algo = SGD_FIM(prngkey_estim, 2000, algo_settings)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(params0.mu1),
        sd=0.001,
        size=model.N,
        likelihood=model.likelihood_array,
        name="slope",
    )
    algo.latent_variables["slope"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False)

    for var in algo.latent_variables.values():
        sdgplt.plot(var)

    return res if save_all else GDResults.make_it_lighter(res)


multi_estim = MultiRunRes(
    [
        one_estim(jrd.PRNGKey(key), myModel, myobs, lbd=0.1, save_all=True)
        for key in range(5)
    ]
)

print(multi_estim.chrono)

# === PLOT === #

names = myModel.params_names

sdgplt.plot(multi_estim, dim_ld=4, params_star=p_star, params_names=names)

# sdgplt.plot_mcmc(algo.mcmc)
