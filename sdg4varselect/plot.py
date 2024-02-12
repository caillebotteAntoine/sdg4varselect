"""
Module that define some ploting function.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

from functools import wraps
import matplotlib.pyplot as plt
from sdg4varselect.outputs_new import GDResults, MultiRunRes, RegularizationPathRes

# import matplotlib.colors as colors
import numpy as np
import jax.numpy as jnp


FIGSIZE = 15


def figure():
    fig = plt.figure()
    fig.set_figheight(FIGSIZE)
    fig.set_figwidth(FIGSIZE)
    return fig


def plot_sample(obs, sim, params_star, censoring_loc, a, b):

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

    # print(f'censoring = {int((1-obs["delta"].mean())*100)}%')

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


def dim_standardize(list_x: list):
    max_row = max([x.shape[0] for x in list_x])
    max_col = max([x.shape[1] for x in list_x])
    return jnp.array(
        [
            jnp.pad(
                x,
                ((0, max_row - x.shape[0]), (0, max_col - x.shape[1])),
                constant_values=None,
            )
            for x in list_x
        ]
    )


def plot(*args, **kwargs):
    x = args[0]
    if isinstance(x, (GDResults, MultiRunRes)):
        if "dim_ld" in kwargs:
            return plot_theta(*args, **kwargs), plot_theta_hd(*args, **kwargs)
        return plot_theta(*args, **kwargs)

    if isinstance(x, RegularizationPathRes):
        return plot_reg_path(*args, **kwargs)

    return plt.plot(*args, **kwargs)


# ===================================================== #
def _plot_theta(multi_theta, dim_ld=None, params_star=None, params_names=None):
    dt = dim_standardize(multi_theta).T

    if dim_ld is None:
        dim_ld = dt.shape[0]

    if params_names is None:
        params_names = [str(i) for i in range(dim_ld)]

    if params_star is not None:
        params_star = jnp.hstack(params_star)

    fig = figure()
    for i in range(dim_ld):
        ax = fig.add_subplot(dim_ld, 1, i + 1)

        ax.plot(dt[i], label=params_names[i])

        if params_star is not None:
            ax.axhline(
                params_star[i], linestyle="--", label=params_names[i], color=f"C{i}"
            )

        ax.legend(loc="center left")

    if dim_ld != 0:
        ax = fig.axes[0]
        ax.set_title("Parameter")

    return fig, fig.axes


def plot_theta(multi_estim, dim_ld=None, params_star=None, params_names=None):
    if isinstance(multi_estim, RegularizationPathRes):
        return plot_theta(
            multi_estim[multi_estim.argmin_bic],
            dim_ld,
            params_star,
            params_names,
        )

    elif isinstance(multi_estim, MultiRunRes):
        multi_theta = [res.theta for res in multi_estim]

    elif isinstance(multi_estim, GDResults):
        multi_theta = [multi_estim.theta]
    else:
        raise TypeError("multi_estim must be MultiRunRes or GDResults")

    return _plot_theta(multi_theta, dim_ld, params_star, params_names)


def plot_theta_hd(multi_estim, dim_ld=None, params_star=None, params_names=None):
    if isinstance(multi_estim, RegularizationPathRes):
        return plot_theta_hd(
            multi_estim[multi_estim.argmin_bic],
            dim_ld,
            params_star,
            params_names,
        )

    elif isinstance(multi_estim, MultiRunRes):
        multi_theta = [res.theta for res in multi_estim]

    elif isinstance(multi_estim, GDResults):
        multi_theta = [multi_estim.theta]
    else:
        raise TypeError("multi_estim must be MultiRunRes or GDResults")

    multi_theta = [theta[:, dim_ld:] for theta in multi_theta]

    if params_star is not None:
        params_star = jnp.hstack(params_star)[dim_ld:]
    if params_names is not None:
        params_names = params_names[dim_ld:]

    return _plot_theta(multi_theta, multi_theta[0].shape[-1], params_star, params_names)


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


def plot_reg_path(reg_res: RegularizationPathRes, dim_ld: int = 0):
    if isinstance(reg_res, RegularizationPathRes):
        multi_theta_hd = [res[-1].last_theta[dim_ld:] for res in reg_res.multi_run]
        lbd_set = reg_res.lbd_set
        bic = reg_res.bic[-1]

        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Regularization path")
        ax.set_xlabel(r"Regularization penalty ($\lambda$)")
        ax.set_ylabel(r"HD Parameter ($\beta$)")

        ax.set_xscale("log")

        ax.plot(lbd_set, multi_theta_hd)

        ax_bic = ax.twinx()
        ax_bic.plot(lbd_set, bic, color="k", linewidth=5, linestyle="-")
        ax_bic.plot(lbd_set, bic, color="w", linewidth=3, linestyle="-")
        ax_bic.plot(lbd_set, bic, color="k", linewidth=2, linestyle="--", label="BIC")
        ax_bic.set(ylabel="BIC Score")
        ax_bic.set(ylabel="BIC Score")

        # minimum value of bic
        ax_bic = plot_axvline(
            ax_bic, lbd_set, bic, reg_res.argmin_bic, color="b", msg=" = min(BIC)"
        )

        return fig, [ax, ax_bic]


def plot_box_plot_hd(theta, dim_ld=0, params_star=None, threshold=0):

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)

    multi_theta = jnp.array([t[dim_ld:] for t in theta]).T
    num_support = (multi_theta != 0).sum(axis=1)

    id_support = np.array(
        [i for i in range(len(num_support)) if num_support[i] >= threshold]
    )
    xticks = [i + 1 for i in range(len(id_support))]

    ax.boxplot(multi_theta[:, id_support])

    if params_star is not None:
        params_star = jnp.hstack(params_star)[dim_ld:]
        ax.plot(xticks, params_star[id_support], "bs", label="true value")
        ax.set_xticks(xticks, id_support)

        ax.legend()

    return fig, ax


def plot_mcmc(x, id_max=None):
    """plot an MCMC_chain"""

    if id_max is None:
        id_max = len(x.chain)
    if len(x.sd) == 1:
        fig, axs = plt.subplots(2, 1, sharex=True)
    else:
        fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].set_title(label="chaine de " + x.name)

    axs[0].plot(x.chain[:id_max])
    axs[0].set_ylabel("chain")

    axs[1].plot(x.acceptance_rate()[:id_max])
    axs[1].set_ylabel("acceptance_rate")

    if len(x.sd) != 1:
        axs[2].plot(x.sd[:id_max])
        axs[2].set_ylabel("proposal sd")

    axs[-1].set_xlabel("Iteration")

    return fig, axs
