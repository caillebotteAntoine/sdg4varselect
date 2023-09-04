# Create by antoine.caillebotte@inrae.fr
from time import time
from sdg4varselect.miscellaneous import time2string

import pickle
import numpy as np

from one_run import (
    kwargs_run_GD,
    sample,
    estim,
    get_random_params0,
    params_star_weibull,
    params_star_stack,
    N_IND,
    DIM_COV,
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
        res, solver, key = estim(data_set, parameters0, key, verbatim=verbatim)
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


# ====================================================== #
# ================ REGULARIZATION PATH ================= #
# ====================================================== #
params0, prng_key = get_random_params0(jrd.PRNGKey(123), error=0.2)
# params0["mu1"] = params_star_weibull.mu1
# params0["mu2"] = params_star_weibull.mu2
# params0["mu3"] = params_star_weibull.mu3
# params0["gamma2_1"] = params_star_weibull.gamma2_1
# params0["gamma2_2"] = params_star_weibull.gamma2_2

# params0["alpha"] = params_star_weibull.alpha

print(f"params0 = {params0}")

# pour DIM_COV < 100
lbd_set = 10 ** jnp.linspace(-1, -0.5, num=50)  # [10**-0.75]  #
# pour DIM_COV = 1000
lbd_set = 10 ** jnp.linspace(-2, -0.5, num=50)  #


nrun = 10
ls, lr = [], []
for k in range(nrun):
    time_start = time()
    s, r, prng_key = clever_regularization_path(params0, lbd_set, prng_key)
    print(f"REGULARIZATION PATH TIME: {time2string(time() - time_start)}s")

    ls.append(s)
    lr.append(r)


bic, theta_reg = bic_final_estim_from_list(ls, N_IND, DIM_COV)
# # ====================================================== #
# # ====================================================== #
step_size = {
    "jac": ls[0][0].step_size,
    "fisher": ls[0][0].step_size_fisher,
    "gradient": ls[0][0].step_size_grad,
}

bic_argmin = np.argmin(bic)
latent_variables = ls[0][bic_argmin].latent_variables
for var in latent_variables.values():
    var.likelihood = None

res_selection = lr[0][bic_argmin]

det_fim = [jnp.linalg.det(x) for x in res_selection.fisher_info]
vp_fim = np.array([jnp.linalg.eigvalsh(x) for x in res_selection.fisher_info])


res = {
    "theta": res_selection.theta,
    "grad_precond": res_selection.grad_precond,
    "likelihood": res_selection.likelihood,
    "theta_diff": res_selection.theta_diff,
    "jac_min": [res_selection.jac[i].min() for i in range(len(res_selection.jac))],
    "jac_max": [res_selection.jac[i].max() for i in range(len(res_selection.jac))],
    "fim_det": det_fim,
    "fim_vp": vp_fim,
}

data = {
    "res_selection": res,
    "bic": bic,
    "theta_reg": theta_reg,
    "lbd_set": lbd_set,
    "params_names": ls[0][0].params_names,
    "latent_variables": ls[0][bic_argmin].latent_variables,
    "step_size": step_size,
    "DIM_COV": DIM_COV,
    "N_IND": N_IND,
    "params_star_stack": params_star_stack,
}

with open("res_selection.pkl", "wb") as f:
    pickle.dump(data, f)

print("RESULT SAVED !")
