# Create by antoine.caillebotte@inrae.fr

from collections import namedtuple
from functools import partial
from time import time

from random import uniform

import matplotlib.pyplot as plt
import numpy as np
import parametrization_cookbook.jax as pc
import pandas

import sdg4varselect.plot as sdgplt
from joint_model.data_generation import data_simulation
from sdg4varselect import Algorithm, jacrev, jit, jnp, jrd, learning_rate, print_array
from sdg4varselect.logistic_model import gaussian_prior, logistic_curve
from sdg4varselect.miscellaneous import step_message
from sdg4varselect.solver import shrink_support

# ========================================================= #
# ==================== DATA GENERATION ==================== #
# ========================================================= #
DIM_COV = 10

beta = np.zeros(shape=(DIM_COV,))
beta[0] = -2
if DIM_COV > 1:
    beta[1] = -1
if DIM_COV > 2:
    beta[2] = 1
if DIM_COV > 3:
    beta[3] = 2

params_weibull = namedtuple(
    "params_weibull",
    ("mu1", "mu2", "mu3", "gamma2_1", "gamma2_2", "sigma2", "a", "b", "alpha", "beta"),
)

params_star_weibull = params_weibull(
    mu1=0.3,
    mu2=90.0,
    mu3=7.5,
    gamma2_1=0.0025,
    gamma2_2=20,
    sigma2=0.001,
    a=80.0,
    b=35,
    alpha=11.11,
    beta=beta,
)

N_IND = 100
obs, sim, _ = data_simulation(
    params=params_star_weibull,
    key=jrd.PRNGKey(456),
    N=N_IND,
    J=20,
    p=DIM_COV,
    t_min=60,
    t_max=135,
    cov_law="uniform",  # between -1 and 1
)


fig, axs = plt.subplots(1, 2)
axs[0].plot(obs["time"], obs["Y"].T)
axs[0].set_title(label="Longitudinal data")
axs[1].hist(obs["T"], bins=20)
axs[1].set_title(label="Survival data")


print(f'phi1.mean = {sim["phi1"].mean()}, phi1.var = {sim["phi1"].var()}')
print(f'phi2.mean = {sim["phi2"].mean()}, phi2.var = {sim["phi2"].var()}')
# ============================================================= #
# ==================== END DATA GENERATION ==================== #
# ============================================================= #
# ====================== PARAMIETRIZATION ===================== #
# ============================================================= #

parametrization = pc.NamedTuple(
    mu1=pc.RealPositive(scale=0.5),
    mu2=pc.Real(scale=100),
    mu3=pc.RealPositive(scale=10),
    gamma2_1=pc.RealPositive(scale=0.001),
    gamma2_2=pc.RealPositive(scale=10),
    sigma2=pc.RealPositive(scale=0.001),
    alpha=pc.Real(scale=10),
    beta=pc.Real(scale=1, shape=(DIM_COV,)),
)
params = namedtuple(
    "params",
    (
        "mu1",
        "mu2",
        "mu3",
        "gamma2_1",
        "gamma2_2",
        "sigma2",
        "alpha",
        "beta",
    ),
)
params_star = params(
    mu1=params_star_weibull.mu1,
    mu2=params_star_weibull.mu2,
    mu3=params_star_weibull.mu3,
    gamma2_1=params_star_weibull.gamma2_1,
    gamma2_2=params_star_weibull.gamma2_2,
    sigma2=params_star_weibull.sigma2,
    alpha=params_star_weibull.alpha,
    beta=beta,
)
params_star_stack = jnp.hstack(params_star)

# ==================================================== #
# ==================== LIKELIHOOD ==================== #
# ==================================================== #


@jit
def log_hazard(
    time: jnp.ndarray,  # shape = (N,num)
    phi1: jnp.ndarray,  # shape = (N,)
    phi2: jnp.ndarray,  # shape = (N,)
    mu3: jnp.ndarray,  # shape = (1,)
    a: jnp.ndarray,  # shape = (1,)
    b: jnp.ndarray,  # shape = (1,)
    beta: jnp.ndarray,  # shape = (p,)
    alpha: jnp.ndarray,  # shape = (1,)
    cov: jnp.ndarray,  # shape = (N,p)
) -> jnp.ndarray:  # shape = (N, num)
    """hazard(t) = h0(t) * exp(beta^T U  + alpha*m(t))
    with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

    return : log(b/a) + (b-1)*log(t/a) + beta^T U + alpha*m(t)
    """

    logistic_value = logistic_curve(time, phi1, phi2, jnp.array([mu3]))
    assert logistic_value.shape == time.shape

    log_h_0 = jnp.log(b / a)
    log_h_0 += (b - 1) * jnp.log(time / a)
    assert log_h_0.shape == time.shape

    beta_prod_cov = (cov @ beta)[:, None]
    assert beta_prod_cov.shape[0] == log_h_0.shape[0]

    out = log_h_0 + alpha * logistic_value
    return beta_prod_cov + out


@jit
def likelihood_survival_without_prior(
    params, phi1, phi2, T, cov, **kwargs
) -> jnp.ndarray:
    """return likelihood without the gaussian prior"""
    (N,) = T.shape
    (p,) = params.beta.shape
    assert T.shape == (N,)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)
    assert cov.shape == (N, p)
    # ===================== #
    # === survival_likelihood === #
    # ===================== #
    # survival_likelihood = log(survival_fct) + log(hazard_fct)

    # ================= survival_fct ================= #
    # log_survival_fct = - int_0^T hazard(s) ds
    time_s = jnp.linspace(0, T, num=100)[1:].T

    hazard_kwargs = {
        "time": time_s,
        "phi1": phi1,
        "phi2": phi2,
        "mu3": params.mu3,
        "a": 80,  # params_star_weibull.a,
        "b": 35,  # params_star_weibull.b,
        "alpha": params.alpha,
        "beta": params.beta,
        "cov": cov,
    }
    log_hazard_value = log_hazard(**hazard_kwargs)
    assert time_s.shape == log_hazard_value.shape

    log_survival_fct = -jnp.trapz(jnp.exp(log_hazard_value), time_s)
    assert log_survival_fct.shape == (N,)
    # =============== end survival_fct =============== #

    # ================= hazard_fct ================= #
    # log_hazard_fct = log(b*a^-b * T^{b-1}) + beta^T U + alpha*m(T, phi_g)
    # Comme time_s[:,-1] == T, on peut faire :
    log_hazard_fct = log_hazard_value[:, -1]
    assert log_hazard_fct.shape == (N,)
    # =============== end hazard_fct =============== #

    return log_hazard_fct + log_survival_fct


# ============================================================== #
@jit
def likelihood_nlmem_without_prior(
    params, Y, time, phi1, phi2, **kwargs
) -> jnp.ndarray:
    """return likelihood without the gaussian prior"""
    N, J = Y.shape
    assert time.shape == (J,)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)

    pred = logistic_curve(time, phi1, phi2, jnp.array([params.mu3]))  # shape = (N,J)

    likelihood_nlmem = -J / 2 * jnp.log(2 * jnp.pi * params.sigma2) - (
        (Y - pred) ** 2
    ).sum(axis=1) / (2 * params.sigma2)

    assert likelihood_nlmem.shape == (N,)
    return likelihood_nlmem


# ============================================================== #


@partial(
    jit,
    static_argnums=(1),
)
def likelihood_array(theta_reals1d, parametrization, **kwargs):
    """return likelihood"""
    params = parametrization.reals1d_to_params(theta_reals1d)

    latent_prior = gaussian_prior(
        kwargs["phi1"],
        params.mu1,
        params.gamma2_1,
    ) + gaussian_prior(
        kwargs["phi2"],
        params.mu2,
        params.gamma2_2,
    )

    return (
        latent_prior
        + likelihood_nlmem_without_prior(params, **kwargs)
        + likelihood_survival_without_prior(params, **kwargs)
    )


@partial(
    jit,
    static_argnums=(1),
)
def likelihood(theta_reals1d, parametrization, **kwargs):
    return likelihood_array(theta_reals1d, parametrization, **kwargs).sum()


un_jit_jac_likelihood = jacrev(likelihood_array)


@partial(
    jit,
    static_argnums=(1),
)
def jac_likelihood(theta_reals1d, parametrization, **kwargs):
    return un_jit_jac_likelihood(theta_reals1d, parametrization, **kwargs)


# ============================================================== #
# ==================== Solver configuration ==================== #
# ============================================================== #
def get_solver(parametrization, key, params0, plateau_start, plateau_stop):
    solver = Algorithm(key)
    solver.parametrization = parametrization
    solver.theta_reals1d = params0

    # ============================================================ #
    # ==================== MCMC configuration ==================== #
    # ============================================================ #
    if isinstance(params0, dict):
        mu1 = params0["mu1"]
        mu2 = params0["mu2"]
    else:
        mu1 = params0.mu1
        mu2 = params0.mu2

    solver.add_mcmc(
        float(mu1),
        sd=0.001,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi1",
    )
    solver.latent_variables["phi1"].adaptative_sd = True
    solver.add_mcmc(
        float(mu2),
        sd=5,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi2",
    )
    solver.latent_variables["phi2"].adaptative_sd = True
    # ============================================================ #
    # ==================== END configuration ==================== #
    # ============================================================ #

    solver.add_data(**obs)
    solver.likelihood = likelihood
    solver.add_likelihood_kwargs("time", "Y", "phi1", "phi2", "T", "cov")

    solver.add_data(parametrization=solver.parametrization)
    solver.add_likelihood_kwargs("parametrization")

    solver.step_size = learning_rate(plateau_start, np.log(1e-4), plateau_stop, 0.65)
    return solver


# ================================================= #
# ==================== Results ==================== #
# ================================================= #
def get_res(
    solver,
    plateau_start,
    total_iteration,
    fisher_preconditionner,
    fisher_mask,
    proximal_operator,
    lbd,
    verbatim=False,
    **kwargs,
):
    solver.verbatim = verbatim
    res = solver.stochastic_gradient(
        niter=total_iteration,
        jac_likelihood=jac_likelihood,
        fisher_preconditionner=fisher_preconditionner,
        fisher_mask=fisher_mask,
        smart_start=plateau_start if fisher_preconditionner else 0,
        proximal_operator=proximal_operator,
        prox_regul=lbd,
        p=DIM_COV,
    )

    return res


def get_solver_and_res(
    params0, plateau_start, plateau_stop, fisher_preconditionner, **kwargs_run_GD
):
    solver = get_solver(
        parametrization,
        key=jrd.PRNGKey(int(time())),
        params0=params0,
        plateau_start=plateau_start if fisher_preconditionner else 0,
        plateau_stop=plateau_stop,
    )

    return solver, get_res(
        solver,
        plateau_start=plateau_start,
        fisher_preconditionner=fisher_preconditionner,
        **kwargs_run_GD,
    )


beta0 = np.random.uniform(-1, 1, size=DIM_COV)
params0 = {
    "mu1": 0.5,  # 1
    "mu2": 50.0,  # 2
    "mu3": 3.0,  # 3
    "gamma2_1": 0.00025,  # 4
    "gamma2_2": 2.0,  # 5
    "sigma2": 0.0001,  # 6
    "alpha": 5.0,  # 7
    "beta": beta0,
}
# np.zeros(shape=(DIM_COV,)),  # 8 -> 8 + DIM_COV


fisher_mask = (
    jnp.arange(0, len(params0) + DIM_COV - 1) < len(params0) - 2
)  # -2 car on veut pas de 2 paramètre beta and alpha
# fisher_mask = jnp.array([True for i in range(len(fisher_mask))])

assert len(fisher_mask) == len(params_star_stack)


kwargs_run_GD = {
    "fisher_preconditionner": True,
    "fisher_mask": fisher_mask,
    "lbd": 1 / N_IND,
    "proximal_operator": True,
    "plateau_start": 1000,
    "plateau_stop": 1100,
    "total_iteration": 2000,
}


def regularization_path(path, nrep=1, verbatim=False):
    res_solver = []

    for i in range(len(path)):
        # print(i)
        # print(step_message(i, len(path)), end="\r" if not verbatim else "\n")
        res_solver.append([])

        kwargs_run_GD["lbd"] = path[i]
        kwargs_run_GD["proximal_operator"] = True
        for _ in range(nrep):
            (solver, _) = get_solver_and_res(
                **kwargs_run_GD, params0=params0, verbatim=verbatim
            )

            res_solver[-1].append(solver)

    return res_solver


# ====================================================== #
# ================ REGULARIZATION PATH ================= #
# ====================================================== #
# lbd_set = 10 ** jnp.linspace(-1.5, 1, num=50)

# res_solver = regularization_path(path=lbd_set, nrep=1, verbatim=False)


# fig, _, bic_res = sdgplt.plot_regularization_path(
#     res_solver,
#     lbd_set,
#     p=DIM_COV,
#     N=N_IND,
# )


# ax[0].title.set_fontsize(28)
# ax[0].xaxis.label.set_fontsize(28)
# ax[0].yaxis.label.set_fontsize(28)
# fig.savefig("images/regularization_path.png")

# # sdgplt.plot_selected_component(res_solver, lbd_set, DIM_COV)
# sdgplt.plot_rmse(res_solver, params_star, lbd_set, exclude=["beta"])

# lbd_selection = lbd_set[bic_res["bic"] == bic_res["min"]]
# print(f"regularizatio value selected = {lbd_selection}")
# kwargs_run_GD["lbd"] = lbd_selection

# ====================================================== #
# ====================== INFERENCE ===================== #
# ====================================================== #
kwargs_run_GD["lbd"] = 1.3 * 10**-1

n_run = 1
solver_list = [i for i in range(n_run)]
res_list = [i for i in range(n_run)]
res_select_list = [i for i in range(n_run)]
for i in range(n_run):
    print(i)
    p = params0.copy()
    for key in p:
        p[key] *= uniform(0.9, 1)

    kwargs_run_GD["proximal_operator"] = True
    solver_list[i], res_list[i] = get_solver_and_res(
        **kwargs_run_GD, params0=p, verbatim=False
    )

    solver_select, mask_select = shrink_support(solver_list[i], "beta", DIM_COV)
    p["beta"][not mask_select[-DIM_COV:].all()] = 0
    solver_select.reset_solver()
    solver_select.theta_reals1d = p

    kwargs_run_GD["proximal_operator"] = False
    res_select_list[i] = get_res(solver_select, **kwargs_run_GD, verbatim=False)


solver = solver_list[0]
res = res_list[0]
res_select = res_select_list[0]

id = [0, 3, 5, 6]
fig, ax = sdgplt.plot_params(
    x=res.theta[:, id],
    x_star=np.array(params_star_stack)[id],
    p=0,
    names=solver.params_names[id],
    logscale=False,
)

for i in range(len(ax) - 1):
    ax[i].xaxis.set_tick_params(labelbottom=False)
    ax[i].title.set_fontsize(28)
    ax[i].legend(fontsize=18)

    for label in ax[i].get_yticklabels():
        label.set_fontsize(16)

ax[-1].xaxis.label.set_fontsize(28)
ax[-1].legend(fontsize=18)
for label in ax[-1].get_xticklabels() + ax[-1].get_yticklabels():
    label.set_fontsize(16)

fig.savefig("images/theta_example_run.png")


fig, ax = sdgplt.plot_grad(
    x=res.grad[:, id],
    p=0,
    names=solver.params_names[id],
)

for i in range(len(ax) - 1):
    ax[i].xaxis.set_tick_params(labelbottom=False)
    ax[i].title.set_fontsize(28)
    ax[i].legend(fontsize=18)

    for label in ax[i].get_yticklabels():
        label.set_fontsize(16)

ax[-1].xaxis.label.set_fontsize(28)
ax[-1].legend(fontsize=18)
for label in ax[-1].get_xticklabels() + ax[-1].get_yticklabels():
    label.set_fontsize(16)

fig.savefig("images/grad_example_run.png")


fig, ax = sdgplt.plot_params_hd(res.theta, p=DIM_COV, location="right")
ax.title.set_fontsize(28)
ax.xaxis.label.set_fontsize(28)
ax.yaxis.label.set_fontsize(28)
fig.savefig("images/beta.png")

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)


fig, _ = sdgplt.plot_all_params_grad(
    params=res.theta,
    grad=res.grad,
    params_star=params_star_stack,
    p=DIM_COV,
    names=solver.params_names,
    logscale=True,
)


fig, _ = sdgplt.plot_all_params_grad(
    params=res_select.theta,
    grad=res_select.grad,
    params_star=params_star_stack,
    p=DIM_COV,
    names=solver.params_names,
    mask=mask_select,
    logscale=True,
)


fig, ax = sdgplt.ax_plot_line_with_doted_hline(
    plt.figure(), 2, 1, 2, res.step_size, logscale=False, label="step_size"
)

fig, ax = sdgplt.ax_plot_line_with_doted_hline(
    fig, 2, 1, 1, res.likelihood, logscale=False, label="likelihood"
)

id_run = 4

resf = []
for i in range(n_run):
    resf.append(res_list[i].theta[-1])

resf = np.array(resf)

dt = pandas.DataFrame(resf[:, :-DIM_COV], columns=solver.params_names[:-DIM_COV]).melt()

dt.to_csv(f"resf{id_run}.csv", sep=";")

dt = pandas.DataFrame(
    resf[:, -DIM_COV:], columns=[f"beta_{i}" for i in range(DIM_COV)]
).melt()

dt.to_csv(f"resf_beta{id_run}.csv", sep=";")


resf = []
for i in range(n_run):
    resf.append(res_select_list[i].theta[-1])

resf = np.array(resf)

dt = pandas.DataFrame(resf[:, :-DIM_COV], columns=solver.params_names[:-DIM_COV]).melt()

dt.to_csv(f"resf_select{id_run}.csv", sep=";")

dt = pandas.DataFrame(
    resf[:, -DIM_COV:], columns=[f"beta_{i}" for i in range(DIM_COV)]
).melt()

dt.to_csv(f"resf_select_beta{id_run}.csv", sep=";")
