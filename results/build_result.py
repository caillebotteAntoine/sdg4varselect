import pickle
import gzip

import jax.numpy as jnp

from sdg4varselect.logistic import Logistic_JM, get_params_star

from sdg4varselect.plot import (
    plot_theta,
    plot_reg_path,
    plot_theta_HD,
    plot_box_plot_HD,
    figure,
)

model = Logistic_JM(N=100, J=5, DIM_HD=10)
filename = f"testN_s{0}_N{model.N}_P{model.DIM_HD}_J{model.J}"

R = pickle.load(gzip.open(f"files/{filename}.pkl.gz", "rb"))
res = R["res"]
lbd_set = R["lbd_set"]


# === PLOT === #

selection_res = res[0][0]
reg_path = selection_res.regularization_path


params_star = get_params_star(model.DIM_HD)
params_names = model.params_names

plot_theta(reg_path, model.DIM_LD, params_star, params_names)
plot_reg_path(lbd_set, reg_path, selection_res.bic, model.DIM_HD)

estim_res = [r.estim_res for r in res[0]]
theta = jnp.array([r.theta for r in res[0]])
theta_biased = jnp.array([r.theta_biased for r in res[0]])

plot_theta(estim_res, model.DIM_LD, params_star, params_names)
plot_theta_HD(estim_res, model.DIM_LD, params_star, params_names)
plot_box_plot_HD(theta, model.DIM_LD, params_star)


# ====================================================== #
def theta_box_plot(theta, params_star, params_names, title):
    fig = figure()

    for i in range(7):
        ax = fig.add_subplot(3, 3, 1 + i)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3))
        bp = ax.boxplot(theta[:, i], patch_artist=True)

        for patch in bp["boxes"]:
            patch.set(facecolor=f"C{i}")

        for median in bp["medians"]:
            median.set_color("black")

        ax.axhline(y=params_star[i], color="k", label="true value")

        ax.legend()
        ax.set_title(f"{title} {params_names[i]} ")

    return fig, fig.axes


theta_box_plot(theta_biased, params_star, params_names, "biased")
theta_box_plot(theta, params_star, params_names, "EMV of")
