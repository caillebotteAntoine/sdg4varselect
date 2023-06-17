# Create by antoine.caillebotte@inrae.fr
from time import time
from sdg4varselect.miscellaneous import time2string

from sdg4varselect.solver import shrink_support
from random import uniform
import numpy as np
import pandas

from one_run import (
    kwargs_run_GD,
    sample,
    estim,
    sample_and_estim,
    params0,
    N_IND,
    DIM_COV,
    params_star_stack,
)
from sdg4varselect.miscellaneous import step_message

from sdg4varselect import jrd, jnp
import sdg4varselect.plot as sdgplt


folder = "images"


def regularization_path(path, prng_key, nrep=1, verbatim=False):
    res_solver = []

    for i in range(len(path)):
        print(step_message(i, len(path)), end="\r" if not verbatim else "\n")

        res_solver.append([])

        kwargs_run_GD["prox_regul"] = path[i]
        kwargs_run_GD["proximal_operator"] = True
        for _ in range(nrep):
            _, solver, prng_key = sample_and_estim(params0, prng_key, verbatim=verbatim)

            res_solver[-1].append(solver)

    return res_solver, prng_key


def multi_estim(n_run, prng_key, verbatim=True):
    solver_list, res_list, res_select_list = (
        [None for _ in range(n_run)],
        [None for _ in range(n_run)],
        [None for _ in range(n_run)],
    )

    for i in range(n_run):
        print(step_message(i, n_run), end="\r" if not verbatim else "\n")

        p = params0.copy()
        for key in p:
            key_new, prng_key = jrd.split(prng_key, 2)
            p[key] *= float(jrd.uniform(key_new, minval=0.8))

        print(p)

        kwargs_run_GD["proximal_operator"] = True
        res_list[i], solver_list[i], prng_key = sample_and_estim(
            p, prng_key, verbatim=verbatim
        )

        solver_select, mask_select = shrink_support(solver_list[i], "beta", DIM_COV)
        p["beta"][not mask_select[-DIM_COV:].all()] = 0
        solver_select.reset_solver()
        solver_select.theta_reals1d = p

        kwargs_run_GD["proximal_operator"] = False
        res_select_list[i] = estim(solver_select, verbatim=verbatim)

    return solver_list, res_list, res_select_list


# ====================================================== #
# ================ REGULARIZATION PATH ================= #
# ====================================================== #
lbd_set = 10 ** jnp.linspace(-1, 1, num=50)

time_start = time()
res_solver, prng_key = regularization_path(lbd_set, jrd.PRNGKey(0), verbatim=False)
print(time2string(time() - time_start))

fig, ax, bic_res = sdgplt.plot_regularization_path(
    res_solver,
    lbd_set,
    p=DIM_COV,
    N=N_IND,
)

ax[0].title.set_fontsize(28)
ax[0].xaxis.label.set_fontsize(28)
ax[0].yaxis.label.set_fontsize(28)
fig.savefig(folder + "/regularization_path.png")

print(bic_res)
# ====================================================== #
# ====================== INFERENCE ===================== #
# ====================================================== #
lbd_selection = lbd_set[bic_res["bic"] == bic_res["min"]]
print(f"regularization value selected = {lbd_selection}")
kwargs_run_GD["prox_regul"] = lbd_selection

n_run = 50
time_start = time()
solver_list, res_list, res_select_list = multi_estim(
    n_run, jrd.PRNGKey(0), verbatim=False
)
print(time2string(time() - time_start))

solver = solver_list[0]
res = res_list[0]
res_select = res_select_list[0]

# ====================================================== #
# ====================================================== #
# ====================================================== #
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

fig.savefig(folder + "/theta_example_run.png")
# ====================================================== #


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

fig.savefig(folder + "/grad_example_run.png")
# ====================================================== #


fig, ax = sdgplt.plot_params_hd(res.theta, p=DIM_COV, location="right")
ax.title.set_fontsize(28)
ax.xaxis.label.set_fontsize(28)
ax.yaxis.label.set_fontsize(28)
fig.savefig(folder + "/beta.png")
# ====================================================== #


def save_estimates(res, file_name):
    estimates = np.array([res[i].theta[-1] for i in range(len(res))])

    theta = pandas.DataFrame(
        estimates[:, :-DIM_COV], columns=solver.params_names[:-DIM_COV]
    ).melt()
    beta = pandas.DataFrame(
        estimates[:, -DIM_COV:], columns=[f"beta_{i}" for i in range(DIM_COV)]
    ).melt()

    theta.to_csv(file_name + "theta.csv", sep=";")
    beta.to_csv(file_name + "beta.csv", sep=";")


# ====================================================== #
save_estimates(res_list, folder + "/penalized_estimate_")
save_estimates(res_select_list, folder + "/estimate_")
# ====================================================== #
