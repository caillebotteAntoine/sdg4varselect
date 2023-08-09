# Create by antoine.caillebotte@inrae.fr
from time import time
from sdg4varselect.miscellaneous import time2string

from sdg4varselect.solver import shrink_support
import numpy as np
import pandas
import pickle

from one_run import (
    kwargs_run_GD,
    sample,
    estim,
    estim_solver,
    get_random_params0,
    params_star_weibull,
    N_IND,
    DIM_COV,
    params_star_stack,
)
from sdg4varselect.miscellaneous import step_message, bic_final_estim_from_list

from sdg4varselect import jrd, jnp
import sdg4varselect.plot as sdgplt

folder = "images"


with open("res_selection.pkl", "rb") as f:
    data = pickle.load(f)

list_res = data["list_res"]
bic = data["bic"]
theta_reg = data["theta_reg"]
lbd_set = data["lbd_set"]
params_names = data["params_names"]
latent_variables = data["latent_variables"]
step_size = data["step_size"]

with open("res_selection.pkl", "wb") as f:  # open a text file
    pickle.dump(data, f)

# # ====================================================== #

_, _ = sdgplt.plot_regularization_path(theta_reg, lbd_set, bic, p=DIM_COV)

# # ====================================================== #
# # ====================================================== #

bic_argmin = np.argmin(bic)
print(f"regularization value selected = {lbd_set[bic_argmin]}")

res_selection = list_res[0][bic_argmin]


# # ====================================================== #

fig, ax = sdgplt.plot_params(
    x=res_selection.theta,
    x_star=np.array(params_star_stack),
    p=DIM_COV,
    names=params_names,
    logscale=False,
)

_, _ = sdgplt.plot_grad(x=res_selection.grad_precond, p=DIM_COV, names=params_names)


_, ax = sdgplt.plot_params_hd(res_selection.theta, p=DIM_COV, location="right")

for var in latent_variables.values():
    sdgplt.plot_mcmc(var)


# ====================================================== #


sdgplt.plot_multi_line(
    np.array([[res_selection.jac[i].max() for i in range(len(res_selection.jac))]]).T,
    0,
    title="maximum de la jacobienne",
)

sdgplt.plot_multi_line(
    np.array([res_selection.likelihood]).T,
    0,
    title="valeur de la vraisemblance",
)

det_fim = np.array([[jnp.linalg.det(x) for x in res_selection.fisher_info_shrink]]).T
sdgplt.plot_multi_line(
    det_fim,
    0,
    title="déterminant de la fim",
    logscale=True,
)


vp_fim = np.array(
    [
        jnp.linalg.eigvalsh(res_selection.fisher_info_shrink[i])
        for i in range(len(res_selection.fisher_info_shrink))
    ]
)

sdgplt.ax_plot_list_of_vector(
    sdgplt.figure(),
    1,
    1,
    1,
    vp_fim,
    title="valeur propre de la fim",
    location="right",
)

sdgplt.plot_multi_line(
    np.array([vp_fim.min(axis=1)]).T,
    0,
    title="valeur propre minimal de la fim",
)


# from sdg4varselect.gradient import prox

# beta = solver_selection.theta_reals1d[-DIM_COV:]

# lbd_set_prox = 10 ** jnp.linspace(-2, 2, num=200)
# beta_prox = np.array([prox(beta, 0.001, lbd=lbd) for lbd in lbd_set_prox])


# sdgplt.ax_plot_list_of_vector(
#     sdgplt.figure(),
#     1,
#     1,
#     1,
#     beta_prox,
#     title="opérateur proximal",
#     location="right",
# )

sdgplt.figure()
step_size["jac"].plot(label="Jac step size")
step_size["fisher"].plot(label="FIM step size")
step_size["gradient"].plot(label="gradient step size")
sdgplt.plt.legend()

# # ====================================================== #
# # ====================== INFERENCE ===================== #
# # ====================================================== #


# solver_inference, mask_select = shrink_support(solver_selection, "beta", DIM_COV)
# params0["beta"][not mask_select[-DIM_COV:].all()] = 0
# solver_inference.reset_solver()
# solver_inference.theta_reals1d = params0

# kwargs_run_GD["proximal_operator"] = False

# time_start = time()
# res_inference = estim_solver(solver_inference, verbatim=False)
# print(f"INFERENCE TIME: {time2string(time() - time_start)}")

# # # ====================================================== #
# # # ====================================================== #


# fig, ax = sdgplt.plot_params(
#     x=res_inference.theta,
#     x_star=np.array(params_star_stack),
#     p=DIM_COV,
#     names=solver_inference.params_names,
#     logscale=False,
# )

# fig, ax = sdgplt.plot_params_hd(res_inference.theta, p=DIM_COV, location="right")
