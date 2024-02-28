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


from sdg4varselect.models.linear_me_model import (
    LinearLatentModelH1,
    LinearLatentModelH0,
)


myModelH1 = LinearLatentModelH1(N=100, J=5)


p_star = myModelH1.new_params(
    mu2=0.1,
    gamma2_2=0.1,
    sigma2=0.001,
)


myobsH1, _ = myModelH1.sample(p_star, jrd.PRNGKey(0))

plt.plot(myobsH1["mem_obs_time"].T, myobsH1["Y"].T, "o-")

algo_settings = get_GDFIM_settings(preheating=400, heating=600)


def one_estim(prngkey, model, data, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))
    params0 = model.parametrization.reals1d_to_params(theta0)

    algo = SGD_FIM(prngkey_estim, 2000, algo_settings)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(params0.mu2),
        sd=0.001,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False)

    # for var in algo.latent_variables.values():
    #     sdgplt.plot(var)

    return (res if save_all else GDResults.make_it_lighter(res)), algo


out = [
    one_estim(jrd.PRNGKey(key), myModelH1, myobsH1, save_all=True) for key in range(1)
]

multi_estimH1 = MultiRunRes([o[0] for o in out])
algorithmsH1 = [o[1] for o in out]

sdgplt.plot(
    multi_estimH1, dim_ld=3, params_star=p_star, params_names=myModelH1.params_names
)


#
#
#
#
#
#
#
#
#
#
#
#
#

myModelH0 = LinearLatentModelH0(N=100, J=5)


def one_estim_H0(prngkey, model, data, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))

    algo = SGD_FIM(prngkey_estim, 2000, algo_settings)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        0,
        sd=0.001,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False)

    # for var in algo.latent_variables.values():
    #     sdgplt.plot(var)

    return (res if save_all else GDResults.make_it_lighter(res)), algo


out = [
    one_estim_H0(jrd.PRNGKey(key), myModelH0, myobsH1, save_all=True)
    for key in range(1)
]

multi_estimH0 = MultiRunRes([o[0] for o in out])
algorithmsH0 = [o[1] for o in out]

sdgplt.plot(
    multi_estimH0, dim_ld=2, params_star=p_star, params_names=myModelH0.params_names
)


# sdgplt.plot_mcmc(algo.mcmc)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from random import randint

theta_chainH0 = multi_estimH0[0].theta
phi2_chainH0 = algorithmsH0[0].latent_variables["phi2"].chain
theta_chainH1 = multi_estimH1[0].theta
phi2_chainH1 = algorithmsH1[0].latent_variables["phi2"].chain

rk = [10]
#####################
for j in range(len(phi2_chainH0)):

    paramsH1 = myModelH1.new_params(
        **dict(zip(("mu2", "gamma2_2", "sigma2"), list(theta_chainH1[0, :])))
    )
    theta1 = myModelH1.parametrization.params_to_reals1d(paramsH1)

    paramsH0 = myModelH0.new_params(
        **dict(zip(("gamma2_2", "sigma2"), list(theta_chainH0[0, :])))
    )
    theta0 = myModelH0.parametrization.params_to_reals1d(paramsH0)

    phi_tilde = phi2_chainH0[0] if randint(0, 1) else phi2_chainH1[0]

    likelihood0 = jnp.exp(
        myModelH0.likelihood_array(theta0, phi2=phi_tilde, **(myModelH0._cst | myobsH1))
    )
    likelihood1 = jnp.exp(
        myModelH1.likelihood_array(theta1, phi2=phi_tilde, **(myModelH1._cst | myobsH1))
    )

    print(f"L1 = {likelihood1.sum()} L0 = {likelihood0.sum()}")
    h = 1 - 2 * (likelihood0 / (likelihood1 * rk[-1] + likelihood0))
    r = rk[-1] - 0.1 * h
    r = jnp.where(r >= 0, r, 0)
    rk.append(r)


r = jnp.array(rk[1:])
plt.figure()
plt.plot(r.sum(axis=1))


paramsH0 = myModelH0.new_params(
    **dict(zip(("gamma2_2", "sigma2"), list(theta_chainH0[-1, :])))
)
theta0 = myModelH0.parametrization.params_to_reals1d(paramsH0)

likelihood_marginalH0 = algorithmsH0[0].likelihood_marginal(myModelH0, myobsH1, theta0)


paramsH1 = myModelH1.new_params(
    **dict(zip(("mu2", "gamma2_2", "sigma2"), list(theta_chainH1[0, :])))
)
theta1 = myModelH1.parametrization.params_to_reals1d(paramsH1)

likelihood_marginalH1 = algorithmsH1[0].likelihood_marginal(myModelH1, myobsH1, theta1)
