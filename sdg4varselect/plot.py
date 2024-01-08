import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import jax.numpy as jnp

from functools import wraps


def figure(figsize=15):
    fig = plt.figure()
    fig.set_figheight(figsize)
    fig.set_figwidth(figsize)
    return fig


def plot(*args, **kwargs):
    return plt.plot(*args, **kwargs)


def plot_sample(obs, sim, params_star, censoring_loc, a, b):
    from matplotlib import pyplot as plt

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(7)

    ax = fig.add_subplot(211)
    ax.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

    ax = fig.add_subplot(212)
    ax.hist(
        [obs["T"], sim["T uncensored"]],  # , sim["C"]],
        bins=20,
        density=True,
        label=["censored survival time", "survival time"],  # , "censuring time"],
    )

    def weibull_fct(t, a, b):
        return b / a * (t / a) ** (b - 1) * np.exp(-((t / a) ** b))

    t = np.linspace(
        obs["T"].min(), max(obs["T"].max(), sim["T uncensored"].max()), num=100
    )
    ax.plot(t, weibull_fct(t, a, b), label="weibull baseline")

    ax.plot(
        t,
        weibull_fct(t, censoring_loc, 35),
        label="censured time weibull distribution",
    )
    ax.legend()

    print(f'censoring = {int((1-obs["delta"].mean())*100)}%')

    fig.suptitle(f'Simulation with {int((1-obs["delta"].mean())*100)}% censored data')
    return fig, ax


def dec_figsize(func):
    @wraps(func)
    def new_func(*args, figsize=15, **kwargs):
        out = func(*args, **kwargs)
        if out is not None and len(out) > 0:
            out[0].set_figheight(figsize)
            out[0].set_figwidth(figsize)

        return out

    return new_func


def remove_nan(x):
    return x[~jnp.isnan(x).any(axis=1)]


def dec_log_yscale(func):
    @wraps(func)
    def new_func(*args, logscale=False, **kwargs):
        out = func(*args, **kwargs)
        if out is not None and len(out) > 0:
            if logscale:
                out[1].set_yscale("log")

        return out

    return new_func


# ===================================================== #
def _plot_theta(multi_theta, DIM_LD, params_star, params_names):
    fig = figure()
    ax = fig.add_subplot(DIM_LD, 1, 1)
    ax.set_title("Parameter")
    for i in range(DIM_LD):
        ax = fig.add_subplot(DIM_LD, 1, i + 1)

        ax.plot(multi_theta[i])
        ax.axhline(params_star[i], linestyle="--", label=params_names[i], color=f"C{i}")
        ax.legend(loc="center left")

    return fig, ax


def plot_theta(multi_estim, DIM_LD, params_star, params_names):
    if len(multi_estim) == 0:
        return None, None

    if not isinstance(multi_estim, list):
        multi_estim = [multi_estim]

    multi_theta = jnp.array([res.theta for res in multi_estim]).T
    return _plot_theta(multi_theta, DIM_LD, jnp.hstack(params_star), params_names)


def plot_theta_HD(multi_estim, DIM_LD, params_star, params_names):
    if len(multi_estim) == 0:
        return None, None

    if not isinstance(multi_estim, list):
        multi_estim = [multi_estim]

    multi_theta = jnp.array([res.theta[:, DIM_LD:] for res in multi_estim]).T
    params_star = jnp.hstack(params_star)[DIM_LD:]

    return _plot_theta(
        multi_theta, multi_theta.shape[0], params_star, params_names[DIM_LD:]
    )


def plot_axvline(ax, lbd_set, bic, id, color, msg=""):
    lbd = lbd_set[id]

    ax.axvline(
        x=lbd,
        color=color,
        linewidth=2,
        linestyle="--",
        label=r"$\lambda$" + msg,
    )
    ax.text(
        lbd,
        0.8 * bic.max() + 0.2 * bic.min(),
        rf"$\lambda$ = {lbd_set[id]:.3e}",
        ha="center",
        va="center",
        rotation="vertical",
        backgroundcolor="white",
    )

    return ax


def plot_reg_path(lbd_set, reg_path, bic, DIM_HD):
    multi_theta = jnp.array([res.theta for res in reg_path])
    multi_theta_HD = multi_theta[:, -1, -DIM_HD:]

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Regularization path")
    ax.set_xlabel(r"Regularization penalty ($\lambda$)")
    ax.set_ylabel(r"HD Parameter ($\beta$)")
    ax.set_xscale("log")

    ax.plot(lbd_set, multi_theta_HD)

    jnp.array([multi_theta_HD != 0]).sum(axis=1)

    ax_bic = ax.twinx()
    ax_bic.plot(lbd_set, bic, color="k", linewidth=2, linestyle="--", label="BIC")

    # minimum value of bic
    id_min = jnp.nanargmin(bic)
    ax_bic = plot_axvline(ax_bic, lbd_set, bic, id_min, color="b", msg="min(BIC)")

    return fig, [ax, ax_bic]


def plot_box_plot_HD(multi_estim, DIM_LD, params_star, threshold=0):
    params_star = jnp.hstack(params_star)[DIM_LD:]
    multi_theta = jnp.array([res.theta[-1, DIM_LD:] for res in multi_estim]).T

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)

    num_support = (multi_theta != 0).sum(axis=1)

    id = np.array([i for i in range(len(num_support)) if num_support[i] >= threshold])
    xticks = [i + 1 for i in range(len(id))]

    ax.boxplot(multi_theta[:, id])
    ax.plot(xticks, params_star[id], "bs", label="true value")
    ax.set_xticks(xticks, id)
    ax.legend()

    return fig, ax
