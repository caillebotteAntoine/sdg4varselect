from one_run import (
    get_random_params0,
    params_star_weibull,
    params_star_stack,
    N_IND,
    DIM_COV,
    J_OBS,
)
from datetime import datetime
import pickle
import sdg4varselect.plot as sdgplt
from sdg4varselect import jrd, jnp
from time import time
from one_res import clever_regularization_path, final_estim
import numpy as np

from sdg4varselect.miscellaneous import step_message, bic_final_estim_from_list

from sdg4varselect.miscellaneous import time2string

lbd_set = 10 ** jnp.linspace(-2, 0, num=15)


def method(nrun=1, verbatim=True):
    params0, prng_key = get_random_params0(jrd.PRNGKey(int(time())), error=0.2)

    # ====================================================== #
    # ================ REGULARIZATION PATH ================= #
    # ====================================================== #
    ls, lr = [], []
    for k in range(nrun):
        time_start = time()
        s, r, prng_key = clever_regularization_path(
            params0, lbd_set, prng_key, verbatim=verbatim
        )
        print(f"REGULARIZATION PATH TIME: {time2string(time() - time_start)}")

        ls.append(s)
        lr.append(r)

    bic, ebic, theta_reg = bic_final_estim_from_list(
        ls, lr, N_IND, DIM_COV, verbatim=verbatim
    )

    # ============================================ #
    # ================ INFERENCE ================= #
    # ============================================ #
    bic_argmin = np.argmin(bic)
    solver_selection = ls[0][bic_argmin]

    final_res, _ = final_estim(
        solver_selection, params0, lbd_set[bic_argmin], verbatim=False
    )

    return final_res, bic, ebic, theta_reg, lbd_set[bic_argmin], solver_selection


lr = []
llbd = []
lbic = []
lebic = []
ltheta_reg = []


print(f"n = {N_IND}, p = {DIM_COV}, J = {J_OBS}")
print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

nrun = 10
for i in range(nrun):
    print(step_message(i, nrun))

    res, bic, ebic, theta_reg, lbd_select, solver = method(verbatim=False)
    lr.append(res)
    llbd.append(lbd_select)
    lbic.append(bic)
    lebic.append(ebic)
    ltheta_reg.append(theta_reg)

print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


theta = [res.theta[-1] for res in lr]

data = {
    "theta": theta,
    "lbd_select": llbd,
    "params_names": solver.params_names,
    "lbic": lbic,
    "lebic": lebic,
    "ltheta_reg": ltheta_reg,
    "lbd_set": lbd_set,
}

with open("res_multi_run.pkl", "wb") as f:
    pickle.dump(data, f)
