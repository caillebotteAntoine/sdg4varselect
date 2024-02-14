"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd

# import jax.numpy as jnp

from sdg4varselect.outputs import GDResults, MultiRunRes

from sdg4varselect.algo import SPGD_FIM, GradFimSettings


algo_settings = GradFimSettings(
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


def estim(prngkey, model, data, theta0, lbd=None, alpha=1.0):
    params0 = model.parametrization.reals1d_to_params(theta0)

    algo = SPGD_FIM(prngkey, 2000, algo_settings, lbd=lbd, alpha=alpha)
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
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False)

    return res, algo


def one_estim(prngkey, model, dh, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    res_estim, _ = estim(prngkey_estim, model, dh, theta0, lbd=lbd, alpha=alpha)

    return res_estim if save_all else GDResults.make_it_lighter(res_estim)


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
    import sdg4varselect.plot as plt

    names = myModel.params_names

    plt.plot(multi_estim, 7, p_star, names)
    plt.plot(multi_estim, 7, p_star, names)

    # sdgplt.plot_mcmc(algo.mcmc)
