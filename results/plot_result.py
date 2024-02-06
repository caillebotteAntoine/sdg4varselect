import jax.numpy as jnp

from sdg4varselect.models.logistic_joint_model import Logistic_JM, get_params_star

import sdg4varselect.plot as sdgplt
from sdg4varselect.outputs import sdg4vsResults, MultiRunRes

N = (100,)
P = (10,)  # 50, 200, 400, 600, 1000)
C = (0,)
S = 2


def read(n, p, c, s=2):
    """read files results for n and p as parameter"""
    model = Logistic_JM(N=n, J=5, DIM_HD=p)
    return sdg4vsResults.load(model, "files", f"FR_s{s}_C{c}")


if len(N) == 1 and len(P) == 1:
    results = MultiRunRes.new_from_list([read(N[0], P[0], c) for c in C])
    scenarios_labels = [f"C = {c}" for c in C]
if len(N) == 1:
    results = MultiRunRes.new_from_list([read(N[0], p, C[0]) for p in P])
    scenarios_labels = [f"N = {n}" for n in N]
elif len(P) == 1:
    results = MultiRunRes.new_from_list([read(P[0], n, C[0]) for n in N])
    scenarios_labels = [f"P = {p}" for p in P]


# results, lbd_set, N, P = read_multi_files(N=(50, 100, 200, 300), P=(200,))


# === PLOT === #
n_scenario = len(results.MultiRun)
lbd_set = results.lbd_set

for i in range(n_scenario):
    N = results.N[i]
    P = results.P[i]
    C = results.C[i][0]

    myModel = Logistic_JM(N=N, J=5, DIM_HD=P)
    params_star = get_params_star(myModel.DIM_HD)
    params_names = myModel.params_names
    # ==================================================#
    scenarios = results.MultiRun[i]

    # res = scenarios.listSDGResults
    # selection_res = scenarios
    # estim_res = [r.estim_res for r in res]
    # theta = jnp.array([r.theta for r in res])
    # theta_biased = jnp.array([r.theta_biased for r in res])

    # sdgplt.plot_theta(estim_res, myModel.DIM_LD, params_star, params_names)
    # sdgplt.plot_theta_HD(estim_res, myModel.DIM_LD, params_star, params_names)
    # sdgplt.plot_box_plot_HD(theta, myModel.DIM_LD, params_star)

    for i, run in enumerate(scenarios):
        reg_path = run.regularization_path
        bic = run.bic[-1]

        # sdgplt.plot_theta(reg_path, model.DIM_LD, params_star, params_names)
        sdgplt.plot_reg_path(lbd_set, reg_path, bic, myModel.DIM_LD)

    # ====================================================== #
