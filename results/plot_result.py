import jax.numpy as jnp


from sdg4varselect.models.wcox_mem_joint_model import (
    get_params_star,
    create_logistic_weibull_jm,
)

import sdg4varselect.plot as sdgplt
from sdg4varselect.outputs import TestResults


myModel = create_logistic_weibull_jm(50, 5, 5)
x1 = TestResults.load(myModel, root="files", filename_add_on=f"C{0}_S{2}")
myModel = create_logistic_weibull_jm(20, 5, 5)
x2 = TestResults.load(myModel, root="files", filename_add_on=f"C{0}_S{2}")


results = TestResults(
    [x1, x2],
    [
        {"N": 20, "J": 5, "P": 5, "C": 0},
        {"N": 50, "J": 5, "P": 5, "C": 0},
    ],
)


# === PLOT === #
scenarios_labels = results.get_scenarios_labels("N")
n_scenario = results.ntest
lbd_set = results[0][0].lbd_set


myModel = create_logistic_weibull_jm(**results.config[0])
p_star = get_params_star(myModel)
params_names = myModel.params_names


theta = results.last_theta

fig = sdgplt.figure()
for i in range(7):
    ax = fig.add_subplot(3, 3, 1 + i)
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    bp = ax.boxplot(theta[:, :, -1, i], patch_artist=True, labels=scenarios_labels)

    for patch in bp["boxes"]:
        patch.set(facecolor=f"C{i}")

    for median in bp["medians"]:
        median.set_color("black")

    ax.axhline(y=p_star[i], color="k", label="true value")

    ax.legend()
    ax.set_title(f"EMV of {params_names[i]} ")


# N = (100,)
# P = (10,)  # 50, 200, 400, 600, 1000)
# C = (0,)
# S = 2


# def read(n, p, c, s=2):
#     """read files results for n and p as parameter"""
#     model = Logistic_JM(N=n, J=5, DIM_HD=p)
#     return sdg4vsResults.load(model, "files", f"FR_s{s}_C{c}")


# if len(N) == 1 and len(P) == 1:
#     results = MultiRunRes.new_from_list([read(N[0], P[0], c) for c in C])
#     scenarios_labels = [f"C = {c}" for c in C]
# if len(N) == 1:
#     results = MultiRunRes.new_from_list([read(N[0], p, C[0]) for p in P])
#     scenarios_labels = [f"N = {n}" for n in N]
# elif len(P) == 1:
#     results = MultiRunRes.new_from_list([read(P[0], n, C[0]) for n in N])
#     scenarios_labels = [f"P = {p}" for p in P]


# # results, lbd_set, N, P = read_multi_files(N=(50, 100, 200, 300), P=(200,))


# # === PLOT === #
# n_scenario = len(results.MultiRun)
# lbd_set = results.lbd_set

# for i in range(n_scenario):
#     N = results.N[i]
#     P = results.P[i]
#     C = results.C[i][0]

#     myModel = Logistic_JM(N=N, J=5, DIM_HD=P)
#     params_star = get_params_star(myModel.DIM_HD)
#     params_names = myModel.params_names
#     # ==================================================#
#     scenarios = results.MultiRun[i]

#     # res = scenarios.listSDGResults
#     # selection_res = scenarios
#     # estim_res = [r.estim_res for r in res]
#     # theta = jnp.array([r.theta for r in res])
#     # theta_biased = jnp.array([r.theta_biased for r in res])

#     # sdgplt.plot_theta(estim_res, myModel.DIM_LD, params_star, params_names)
#     # sdgplt.plot_theta_HD(estim_res, myModel.DIM_LD, params_star, params_names)
#     # sdgplt.plot_box_plot_HD(theta, myModel.DIM_LD, params_star)

#     for i, run in enumerate(scenarios):
#         reg_path = run.regularization_path
#         bic = run.bic[-1]

#         # sdgplt.plot_theta(reg_path, model.DIM_LD, params_star, params_names)
#         sdgplt.plot_reg_path(lbd_set, reg_path, bic, myModel.DIM_LD)

#     # ====================================================== #
