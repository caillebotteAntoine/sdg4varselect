import jax.numpy as jnp
import jax.random as jrd

import gzip
import pickle

import sdg4varselect.plot as sdgplt
from sdg4varselect.algo import NanError, estim_res, SPG_FIM
from sdg4varselect.miscellaneous import step_message
from sdg4varselect.logistic import Logistic_JM
from sdg4varselect.data_handler import DataHandler


model = Logistic_JM(N=100, J=5, DIM_HD=100)

# Simulation parameter
params_star = model.new_params(
    mu1=0.3,
    mu2=90.0,
    mu3=7.5,
    gamma2_1=0.0025,
    gamma2_2=20,
    sigma2=0.001,
    alpha=11.11,
    beta=jnp.concatenate(
        [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(model.DIM_HD - 4,))]
    ),
)

obs, sim = model.sample(params_star, jrd.PRNGKey(0), weibull_censoring_loc=7700)

dh = DataHandler()
dh.add_data(**obs)

_ = sdgplt.plot_sample(obs, sim, params_star, 77, 80, 35)


algo_settings = SPG_FIM.settings(
    step_size_grad={
        "learning_rate": 1e-8,
        "preheating": 400,
        "heating": 600,
        "max": 0.9,
    },
    step_size_approx_sto={
        "learning_rate": 1e-8,
        "preheating": 400,
        "heating": None,
        "max": 1,
    },
    step_size_fisher={
        "learning_rate": 1e-8,
        "preheating": 400,
        "heating": None,
        "max": 0.9,
    },
    max_iter=2000,
)


def estim(PRNGKey, model, dh, theta0, lbd=None, alpha=1.0):
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


def selection_then_estimation(PRNGKey, model, dh, lbd=None, alpha=1.0, save_all=True):
    PRNGKey_selection, PRNGKey_estimation = jrd.split(PRNGKey)
    # === SELECTION === #
    res_first_estim = one_estim(PRNGKey_selection, model, dh, lbd, save_all=save_all)

    # === ESTIMATION === #
    DIM_LD = model.DIM_LD
    theta_biased = res_first_estim.theta[-1, DIM_LD:]
    selected_component = theta_biased != 0
    NEW_DIM_HD = int(selected_component.sum())

    model_shrink = Logistic_JM(N=model.N, J=model.J, DIM_HD=NEW_DIM_HD)
    dh_shrink = dh.deepcopy()
    dh_shrink.data["cov"] = dh.data["cov"][:, selected_component]

    res_second_estim = one_estim(
        PRNGKey_estimation, model_shrink, dh_shrink, lbd=None, save_all=save_all
    )

    # === THETA RE CONSTRUCTION === #
    id = jnp.concatenate([jnp.repeat(True, model_shrink.DIM_LD), selected_component])

    theta = jnp.zeros(shape=(res_second_estim.theta.shape[0], id.shape[0]))
    theta = theta.at[:, id].set(res_second_estim.theta)
    res_second_estim = estim_res(
        theta=theta,
        FIM=res_second_estim.FIM,
        grad=res_second_estim.grad,
        likelihood=res_second_estim.likelihood,
    )
    return res_first_estim, res_second_estim


def regularization_path(PRNGKey, lbd_set, verbatim=False, *args, **kwargs):
    DIM_LD = model.DIM_LD
    PRNGKey_list = jrd.split(PRNGKey, num=len(lbd_set))

    def iter_estim():
        for i in range(len(lbd_set)):
            if verbatim:
                print(step_message(i, len(lbd_set)), end="\r")

            kwargs["lbd"] = lbd_set[i]
            kwargs["PRNGKey"] = PRNGKey_list[i]
            selection, estimation = selection_then_estimation(*args, **kwargs)

            if (estimation.theta[-1, DIM_LD:] != 0).sum() == 0:
                for k in range(len(lbd_set) - i):
                    yield selection, estimation
                break
            else:
                yield selection, estimation

    return [res for res in iter_estim()]


lbd_set = 10 ** jnp.linspace(-2, 0, num=15)
reg_path = regularization_path(
    jrd.PRNGKey(0),
    lbd_set=lbd_set,
    save_all=False,
    verbatim=True,
    model=model,
    dh=dh,
)

 pickle.dump( {"reg_path": reg_path, "lbd_set": lbd_set},
     gzip.open("new_reg_path_100.pkl.gz", "wb"),
)


import sdg4varselect.plot as sdgplt
from sdg4varselect.logistic import get_params_star

sdgplt.plot_theta([r[1] for r in reg_path], 7, params_star, model.params_names)
