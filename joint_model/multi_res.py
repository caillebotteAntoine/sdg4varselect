from datetime import datetime
import pickle
import sdg4varselect.plot as sdgplt
from sdg4varselect import jrd, jnp
from time import time
from one_res import clever_regularization_path, final_estim
import numpy as np

from sdg4varselect.miscellaneous import step_message, bic_final_estim_from_list

from sdg4varselect.miscellaneous import time2string
from one_run import get_random_params0

lbd_set = 10 ** jnp.linspace(-2, 0, num=15)


def method(params0, N_IND, DIM_COV, J_OBS, CENSORING, nrun=1, verbatim=True):
    prng_key = int(time())
    print(f"prng_key = {prng_key}")
    params0, prng_key = get_random_params0(jrd.PRNGKey(prng_key), params0, error=0.2)

    # ====================================================== #
    # ================ REGULARIZATION PATH ================= #
    # ====================================================== #
    time_start = time()
    res = regularization_path(
        params0,
        params_star_weibull,
        lbd_set,
        N_IND,
        DIM_COV,
        J_OBS,
        CENSORING,
        prng_key=prng_key,
        verbatim=True,
    )
    print(f"REGULARIZATION PATH TIME: {time2string(time() - time_start)}")

    if res != -1:
        ls, lr, prng_key = res
    else:
        prng_key, _ = jrd.split(prng_key, num=2)

    bic, ebic, theta_reg = list_to_BIC(ls, lr, N_IND, DIM_COV, verbatim=verbatim)

    # ============================================ #
    # ================ INFERENCE ================= #
    # ============================================ #
    bic_argmin = np.argmin(bic)
    res_selection = lr[bic_argmin]
    solver_selection = ls[bic_argmin]
    lbd_selection = lbd_set[bic_argmin]

    res = final_estim(solver_selection, params0, lbd_selection, verbatim=False)

    if res != -1:
        final_res = res[0]
        final_solver = res[1]
    else:
        final_res, final_solver = -1, -1

    return (
        final_res,
        res_selection,
        bic,
        ebic,
        theta_reg,
        lbd_selection,
        solver_selection,
    )


lrf = []
lrs = []
llbd = []
lbic = []
lebic = []
ltheta_reg = []


print(f"n = {N_IND}, p = {DIM_COV}, J = {J_OBS}")
print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

nrun = 50
for i in range(nrun):
    # print(step_message(i, nrun))

    res = method(verbatim=False)
    if res != -1:
        res_f, res_s, bic, ebic, theta_reg, lbd_select, solver = res
        lrf.append(res_f)
        lrs.append(res_s)
        llbd.append(lbd_select)
        lbic.append(bic)
        lebic.append(ebic)
        ltheta_reg.append(theta_reg)

print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


theta = [res.theta[-1] for res in lrf]
theta_biais = [res.theta[-1] for res in lrs]

data = {
    "theta": theta,
    "theta_biais": theta_biais,
    "lbd_select": llbd,
    "params_names": solver.params_names,
    "lbic": lbic,
    "lebic": lebic,
    "ltheta_reg": ltheta_reg,
    "lbd_set": lbd_set,
}

with open("res_multi_run.pkl", "wb") as f:
    pickle.dump(data, f)
