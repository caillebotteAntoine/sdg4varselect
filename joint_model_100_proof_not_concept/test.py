# Create by antoine.caillebotte@inrae.fr
from time import time
from sdg4varselect.miscellaneous import time2string

import sdg4varselect.plot as sdgplt
import pickle
import numpy as np

from sdg4varselect.solver import shrink_support
from one_run import (
    kwargs_run_GD,
    sample,
    estim,
    get_random_params0,
    params_star_weibull,
    params_star_stack,
    N_IND,
    DIM_COV,
    estim_solver,
)
from sdg4varselect.miscellaneous import step_message, bic_final_estim_from_list

from sdg4varselect import jrd, jnp


def clever_regularization_path(parameters0, path, prng_key, nrep=1, verbatim=False):
    list_res = []
    list_solver = []
    data_set, key = sample(params_star_weibull, prng_key)

    for i in range(len(path)):
        print(step_message(i, len(path)), end="\r" if not verbatim else "\n")

        kwargs_run_GD["prox_regul"] = path[i]
        kwargs_run_GD["proximal_operator"] = True
        res, solver, key = estim(
            data_set,
            parameters0,
            key,
            niter=500,
            verbatim=verbatim,
            activate_fim=False,
            activate_jac_approx=False,
            # Grad
            scale_grad=0.5,
            plateau_grad=400,
        )
        list_solver.append(solver)
        list_res.append(res)

        print(f"#beta = {solver.get_number_of_nonzero(p=DIM_COV)}")

        if solver.get_number_of_nonzero(p=DIM_COV) == 0:
            for k in range(len(path) - i - 1):
                print(
                    step_message(i + k, len(path)), end="\r" if not verbatim else "\n"
                )
                list_solver.append(list_solver[-1])
                list_res.append(list_res[-1])

            print(f"break at {path[i]}")
            break

    return list_solver, list_res, key


def final_estim(solver, parameters0, prox_regul, verbatim=False):
    kwargs_run_GD["prox_regul"] = prox_regul
    kwargs_run_GD["proximal_operator"] = False
    p0 = parameters0.copy()

    solver_select, mask_select = shrink_support(solver, "beta", DIM_COV)
    p0["beta"] = jnp.where(mask_select[-DIM_COV:], p0["beta"], 0)
    solver_select.reset_solver()
    solver_select.theta_reals1d = p0

    res, solver_select = estim_solver(
        solver_select,
        niter=2000,
        verbatim=verbatim,
        activate_fim=True,
        activate_jac_approx=True,
        lr=1e-8,
        # Grad
        plateau_grad=1200,
        plateau_grad_size=100,
        scale_grad=1,
        # Jac
        plateau_jac=1200,
        plateau_jac_size=1000,
        scale_jac=1,
        # Fim
        plateau_fim=1200,
        plateau_fim_size=1000,
        scale_fim=1,
    )
    return res, solver_select


# ====================================================== #
# ================ REGULARIZATION PATH ================= #
# ====================================================== #
params0, prng_key = get_random_params0(jrd.PRNGKey(int(time())), error=0.2)

params0["mu1"] = params_star_weibull.mu1 + 0.1
# params0["mu2"] = params_star_weibull.mu2
# params0["mu3"] = params_star_weibull.mu3
# params0["gamma2_1"] = params_star_weibull.gamma2_1
# params0["gamma2_2"] = params_star_weibull.gamma2_2

# params0["alpha"] = params_star_weibull.alpha


print(f"params0 = {params0}")

# pour DIM_COV < 100
# lbd_set = 10 ** jnp.linspace(-1, -0.5, num=50)  # [10**-0.75]  #
# pour DIM_COV = 1000
lbd_set = [0.3]  # , 10**-1.5]  # 10 ** jnp.linspace(-2, -0.5, num=50)  #


nrun = 1
ls, lr = [], []
for k in range(nrun):
    time_start = time()
    s, r, prng_key = clever_regularization_path(params0, lbd_set, prng_key)
    print(f"REGULARIZATION PATH TIME: {time2string(time() - time_start)}")

    ls.append(s)
    lr.append(r)


bic, theta_reg = bic_final_estim_from_list(ls, N_IND, DIM_COV)
# ============================================ #
# ================ INFERENCE ================= #
# ============================================ #
bic_argmin = np.argmin(bic)
res_selection = lr[0][bic_argmin]
solver_selection = ls[0][bic_argmin]

final_res, final_solver = final_estim(
    solver_selection, params0, lbd_set[bic_argmin], verbatim=True
)

# _, _ = sdgplt.plot_regularization_path(theta_reg, lbd_set, bic, p=DIM_COV)


# ====================================================== #
def extract_data(res, solver):
    latent_variables = ls[0][bic_argmin].latent_variables
    for var in latent_variables.values():
        var.likelihood = None

    data = {
        "theta": res.theta,
        "grad_precond": res.grad_precond,
        "likelihood": res.likelihood,
        "theta_diff": res.theta_diff,
        "latent_variables": ls[0][bic_argmin].latent_variables,
        "jac_min": [res.jac[i].min() for i in range(len(res.jac))],
        "jac_max": [res.jac[i].max() for i in range(len(res.jac))],
        "fim_det": [jnp.linalg.det(x) for x in res.fisher_info],
        "fim_vp": np.array([jnp.linalg.eigvalsh(x) for x in res.fisher_info]),
    }
    return data


data_selection = extract_data(res_selection, ls[0][bic_argmin])
data_final = extract_data(final_res, final_solver)

step_size = {
    "jac": ls[0][0].step_size,
    "fisher": ls[0][0].step_size_fisher,
    "gradient": ls[0][0].step_size_grad,
}

data = {
    "res_selection": data_selection,
    "res_final": data_final,
    "bic": bic,
    "theta_reg": theta_reg,
    "lbd_set": lbd_set,
    "params_names": ls[0][0].params_names,
    "step_size": step_size,
    "DIM_COV": DIM_COV,
    "N_IND": N_IND,
    "params_star_stack": params_star_stack,
}

with open("res_selection.pkl", "wb") as f:
    pickle.dump(data, f)

print("RESULT SAVED !")


params_names = solver_selection.params_names


# fig = sdgplt.figure()
# solver_selection.step_size.plot(label="Jac step size")
# solver_selection.step_size_fisher.plot(label="FIM step size")
# solver_selection.step_size_grad.plot(label="gradient step size")
# sdgplt.plt.legend()

# _, _ = sdgplt.plot_params(
#     x=res_selection.theta,
#     x_star=np.array(params_star_stack),
#     p=DIM_COV,
#     names=params_names,
#     logscale=False,
# )

# _, _ = sdgplt.plot_grad(x=res_selection.grad_precond, p=DIM_COV, names=params_names)

# __doc__, _ = sdgplt.plot_params_hd(res_selection.theta, p=DIM_COV, location="right")


# =========================#
fig = sdgplt.figure()
final_solver.step_size.plot(label="Jac step size")
final_solver.step_size_fisher.plot(label="FIM step size")
final_solver.step_size_grad.plot(label="gradient step size")
sdgplt.plt.legend()

_, _ = sdgplt.plot_params_grad(
    final_res.theta,
    final_res.grad_precond,
    np.array(params_star_stack),
    p=DIM_COV,
    names=params_names,
    logscale=True,
)

_, _ = sdgplt.plot_params_hd(final_res.theta, p=DIM_COV, location="right")

print(final_res.theta[-1][:DIM_COV])
