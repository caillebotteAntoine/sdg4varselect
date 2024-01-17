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


def read(N):
    model = Logistic_JM(N=N, J=5, DIM_HD=200)
    filename = f"testN_s{0}_N{model.N}_P{model.DIM_HD}_J{model.J}"

    R = pickle.load(gzip.open(f"files/{filename}_small.pkl.gz", "rb"))
    # with open(f"files/{filename}.pkl", "rb") as f:
    #     R = pickle.load(f)
    res = R["res"][0]
    lbd_set = R["lbd_set"]
    print(f"{filename} LOADED")
    return res, lbd_set


N = (50, 100, 200, 300)
results = [read(n) for n in N]
lbd_set = results[0][1]
results = [r[0] for r in results]


# === PLOT === #
res = results[0]

# selection_res = res[0][0]
# reg_path = selection_res.regularization_path


model = Logistic_JM(N=20, J=5, DIM_HD=200)
params_star = get_params_star(model.DIM_HD)
params_names = model.params_names

# plot_theta(reg_path, model.DIM_LD, params_star, params_names)
# plot_reg_path(lbd_set, reg_path, selection_res.bic, model.DIM_HD)

# estim_res = [r.estim_res for r in res]
theta = jnp.array([r.theta for r in res])
theta_biased = jnp.array([r.theta_biased for r in res])

# plot_theta(estim_res, model.DIM_LD, params_star, params_names)
# plot_theta_HD(estim_res, model.DIM_LD, params_star, params_names)
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


# ====================================================== #
lbd_selected = jnp.array([[jnp.argmin(r.bic) for r in res] for res in results])

fig = figure()
ax = fig.add_subplot(1, 1, 1)
bp = ax.boxplot(lbd_selected, patch_artist=True)


# ====================================================== #
def rmse(x, x_star):
    return jnp.sqrt(((x - x_star) ** 2).mean())  # axis=1))


def rrmse(x, x_star):
    """
    \sqrt{1/N \sum_{i=1}^n (x_i-x_{star,i})^2}"""
    return jnp.sqrt(((x / x_star - 1) ** 2).mean())
    return rmse(x, x_star) / jnp.sqrt((x**2).sum())  # axis=1))


def errors(x, x_star, DIM_LD):
    return (
        rrmse(x[:DIM_LD], x_star[:DIM_LD]),
        rmse(x[DIM_LD:], x_star[DIM_LD:]) / jnp.abs(x_star[DIM_LD:]).max(),
    )


res_errors = jnp.array(
    [
        [errors(r.theta, jnp.hstack(params_star), model.DIM_LD) for r in res]
        for res in results
    ]
)


fig = figure()
ax = fig.add_subplot(2, 1, 1)
ax.boxplot(res_errors[:, :, 0], labels=[f"N = {n}" for n in N])
ax.set_xlabel("censorship")
ax.set_title("rrmse of nu")

ax = fig.add_subplot(2, 1, 2)
ax.boxplot(res_errors[:, :, 1], labels=[f"N = {n}" for n in N])
ax.set_xlabel("censorship")
ax.set_title("rrmse of beta")
