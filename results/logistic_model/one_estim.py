import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.algo import SPG_FIM, NanError, estim_res

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
    max_iter=2000,
)

# algo_settings = SPG_FIM.settings(
#     step_size_grad={
#         "learning_rate": 1e-8,
#         "preheating": 400,
#         "heating": 600,
#         "max": 0.9,
#     },
#     step_size_approx_sto={
#         "learning_rate": 1e-8,
#         "preheating": 400,
#         "heating": None,
#         "max": 1,
#     },
#     step_size_fisher={
#         "learning_rate": 1e-8,
#         "preheating": 400,
#         "heating": None,
#         "max": 0.9,
#     },
#     max_iter=2000,
# )


def estim(PRNGKey, model, dh, theta0, lbd=None, alpha=1.0):
    # _, _ = sdgplt.plot_sample(data_set, sim, params_star, 2000, 80, 35)

    params0 = model.parametrization.reals1d_to_params(theta0)
    algo = SPG_FIM(PRNGKey, dh, algo_settings, lbd=lbd, alpha=alpha)
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
    res = algo.fit(
        model.jac_likelihood,
        DIM_HD=model.DIM_HD,
        theta0_reals1d=theta0,
        ntry=5,
        partial_fit=False,
    )

    return res, algo


def one_estim(PRNGKey, model, dh, lbd=None, alpha=1.0, save_all=True):
    PRNGKey_theta, PRNGKey_estim, PRNGKey_likelihoohd = jrd.split(PRNGKey, 3)
    theta0 = 0.2 * jrd.normal(PRNGKey_theta, shape=(model.parametrization.size,))

    try:
        res_estim, algo = estim(PRNGKey_estim, model, dh, theta0, lbd=lbd, alpha=alpha)
    except NanError as err:
        print(err)
        return NanError

    res = algo.labelswitch(res_estim)

    theta = jnp.array([model.reals1d_to_hstack_params(t) for t in res.theta])
    return estim_res(
        theta=theta if save_all else jnp.array([theta[0], theta[-1]]),
        FIM=res.FIM if save_all else None,
        grad=res.grad if save_all else jnp.array([res.grad[0], res.grad[-1]]),
        likelihood=algo.likelihood_marginal(model, PRNGKey_likelihoohd, res.theta[-1]),
    )


if __name__ == "__main__":
    from sdg4varselect.logistic import Logistic_JM, sample_one, get_params_star

    model = Logistic_JM(N=50, J=5, DIM_HD=200)

    dh = sample_one(jrd.PRNGKey(0), model, weibull_censoring_loc=2000)

    multi_estim = [one_estim(jrd.PRNGKey(key), model, dh, lbd=None) for key in range(1)]
    while NanError in multi_estim:
        multi_estim.remove(NanError)

    # === PLOT === #
    from sdg4varselect.plot import (
        plot_theta,
        plot_theta_HD,
    )

    params_star = get_params_star(model.DIM_HD)

    plot_theta(multi_estim, model.DIM_LD, params_star, model.params_names)
    plot_theta_HD(multi_estim, model.DIM_LD, params_star, model.params_names)

    # sdgplt.plot_mcmc(algo.mcmc)
