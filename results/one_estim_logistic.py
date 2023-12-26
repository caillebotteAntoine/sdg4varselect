import jax.random as jrd
import jax.numpy as jnp

from collections import namedtuple

from algo import SPG_FIM
from sdg4varselect.logistic import Logistic_model, sample_one, get_params_star
from work import sdgplt

N_IND = 100
J_OBS = 5
DIM_COV = 4

model = Logistic_model(DIM_COV)


def estim(PRNGKey, theta0, lbd=None, alpha=1.0):
    PRNGKey_sim, PRNGKey_estim = jrd.split(PRNGKey, 2)

    dh = sample_one(PRNGKey_sim, N_IND, J_OBS, DIM_COV, weibull_censoring_loc=2000)
    # _, _ = sdgplt.plot_sample(data_set, sim, params_star, 2000, 80, 35)

    params0 = model.parametrization.reals1d_to_params(theta0)
    # =================== MCMC configuration ==================== #
    dh.add_mcmc(
        float(params0.mu1),
        sd=0.001,
        size=N_IND,
        likelihood=model.likelihood_array,
        name="phi1",
    )
    dh.latent_variables["phi1"].adaptative_sd = True
    dh.add_mcmc(
        float(params0.mu2),
        sd=2,
        size=N_IND,
        likelihood=model.likelihood_array,
        name="phi2",
    )
    dh.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #

    algo = SPG_FIM(PRNGKey_estim, dh, algo_settings, lbd=lbd, alpha=alpha)
    res = algo.fit(
        model.jac_likelihood,
        niter=2000,
        DIM_HD=DIM_COV,
        theta0_reals1d=theta0,
    )

    return res, algo


algo_settings = [
    {"learning_rate": 1e-8, "preheating": 1000, "heating": 1400, "max": 0.9},
    {"learning_rate": 1e-8, "preheating": 1000, "heating": 2000, "max": 1},
    {"learning_rate": 1e-8, "preheating": 1000, "heating": 2000, "max": 0.9},
]

theta0 = model.parametrization.params_to_reals1d(
    mu1=0.5,
    mu2=50.0,
    mu3=3.0,
    gamma2_1=0.00025,
    gamma2_2=2.0,
    sigma2=0.0001,
    alpha=0.01,
    beta=jrd.uniform(jrd.PRNGKey(0), shape=(DIM_COV,), minval=-1, maxval=1),
)

estim, algo = estim(jrd.PRNGKey(0), theta0, lbd=None, alpha=1.0)

res = algo.labelswitch(estim)
res = algo.estim_res(
    theta=jnp.array([model.reals1d_to_hstack_params(t) for t in res.theta]),
    FIM=res.FIM,
    grad=res.grad,
)

params_star = get_params_star(DIM_COV)

fig, ax = sdgplt.plot_params(
    x=res.theta[:-1],
    x_star=jnp.hstack(list(params_star)),
    p=DIM_COV,
    # names=theta.params_names,
    logscale=False,
)

sdgplt.plot_mcmc(algo.mcmc)

beta = res.theta[:, 7:]
fig, ax = sdgplt.plot_params(
    x=beta,
    x_star=params_star.beta,
    p=0,
    logscale=False,
)
