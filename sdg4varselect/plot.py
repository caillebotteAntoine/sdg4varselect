"""
Module that define some ploting function.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116
from copy import copy
from functools import wraps
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import jax.numpy as jnp


from sdg4varselect.outputs import (
    get_all_theta,
    GDResults,
    MultiRunRes,
    RegularizationPathRes,
)
from sdg4varselect._MCMC import MCMC_chain

# import matplotlib.colors as colors


FIGSIZE = 15


def figure(height=None, width=None):
    fig = plt.figure()
    fig.set_figheight(FIGSIZE if height is None else height)
    fig.set_figwidth(FIGSIZE if width is None else width)
    return fig


def get_ax(height=None, width=None):
    fig = figure(height=height, width=width)
    return fig.add_subplot(1, 1, 1)


def plot_sample(obs, sim, censoring_loc, a, b):

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
    def new_func(*args, **kwargs):
        print(args)
        ax = func(*args, **kwargs)
        if isinstance(ax, plt.Axes):
            if kwargs["ylogscale"] and (kwargs["x"] > 0).all():
                ax.set_yscale("log")
        else:
            raise TypeError(
                f"You try to set yscale to 'log' for a non-Axes object ! Type is {type(ax)}"
            )

        return ax

    return new_func


def default_figure(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        if "fig" not in kwargs or kwargs["fig"] is None:
            kwargs["fig"] = figure()

        func(*args, **kwargs)
        return kwargs["fig"]

    return new_func


def add_figure(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        if len(args) > 0 and isinstance(
            args[0], (matplotlib.figure.Figure, matplotlib.figure.SubFigure)
        ):
            kwargs["ax"] = args[0].add_subplot(1, 1, 1)
        elif "fig" in kwargs:
            kwargs["ax"] = kwargs["fig"].add_subplot(1, 1, 1)

        elif len(args) > 0 and isinstance(args[0], matplotlib.pyplot.Axes):
            kwargs["ax"] = args[0]
        elif "ax" in kwargs:
            pass
        else:
            kwargs["ax"] = figure().add_subplot(1, 1, 1)

        func(*args, **kwargs)
        return kwargs["ax"].get_figure()

    return new_func


@default_figure
def scatter_estimation(x, y=None, vline=None, labels=None, nrows=3, ncols=10, fig=None):
    if y is None:
        # y = x
        y = jnp.arange(x.shape[1])

    assert vline is None or len(vline) == x.shape[0]
    assert labels is None or len(labels) == x.shape[0]
    colors = jnp.arange(x.shape[1])

    for i in range(x.shape[0]):
        ax = fig.add_subplot(nrows, ncols, 1 + i)
        ax.get_yaxis().set_visible(False)
        ax.set_title(labels[i] if labels is not None else "")

        _ = ax.scatter(x[i], y, c=colors)
        if vline is not None:
            ax.axvline(x=vline[i], color="k")

    fig.axes[0].get_yaxis().set_visible(True)
    return fig


@add_figure
def myBoxplot(
    ax,
    x,
    hline=None,
    hlabel=None,
    xlabels=None,
    facecolor="w",
    mediancolor="black",
    **kwargs,
):
    assert len(x.shape) <= 2
    assert xlabels is None or isinstance(xlabels, (str, list))

    if len(x.shape) == 2:
        for i in range(x.shape[0]):
            myBoxplot(
                ax=ax,
                x=x[i],
                hline=hline,
                xlabels=None if xlabels is None else xlabels[i],
                facecolor=facecolor,
                positions=[i],
                **kwargs,
            )
        return ax

    bp = ax.boxplot(x, patch_artist=True, labels=[xlabels], **kwargs)

    for patch in bp["boxes"]:
        patch.set(facecolor=facecolor)

    for median in bp["medians"]:
        median.set_color("black")

    if hline is not None:
        ax.axhline(y=hline, color="k", label=hlabel)

    if xlabels is None:
        ax.set_xticks([])
        ax.set_xlabel("")

    return ax


@default_figure
def boxplot_estimation(
    x, hline=None, xlabels=None, title=None, nrows=3, ncols=10, fig=None
):
    assert hline is None or len(hline) == x.shape[0]  # as many hlines as boxplots
    assert (
        xlabels is None or len(xlabels) == x.shape[0] or len(xlabels) == x.shape[-1]
    )  # as many hlines as boxplots or subboxplot
    assert title is None or len(title) == x.shape[0]  # as many hlines as boxplots

    for i in range(x.shape[0]):
        ax = fig.add_subplot(nrows, ncols, 1 + i)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3))
        if title is not None:
            ax.set_title(title[i])

        myBoxplot(
            ax=ax,
            x=x[i].T,  # if len(x[i].shape) == 1 else x[i],
            hline=None if hline is None else hline[i],
            hlabel="true value",
            xlabels=(
                None
                if xlabels is None
                else (xlabels[i] if len(x[i].shape) == 1 else xlabels)
            ),
            facecolor=f"C{i}",
        )

    return fig


def plot(*args, **kwargs):
    x = args[0]
    if isinstance(x, (GDResults, MultiRunRes)):
        if "dim_ld" in kwargs:
            return plot_theta(*copy(args), **copy(kwargs)), plot_theta_hd(
                *copy(args), **copy(kwargs)
            )
        return plot_theta(*args, **kwargs)

    if isinstance(x, RegularizationPathRes):
        return plot_reg_path(*args, **kwargs)

    if isinstance(x, dict):
        for v in x.values():
            plot(v, **kwargs)
        return None

    if isinstance(x, MCMC_chain):
        return plot_mcmc(*args, **kwargs)

    return plt.plot(*args, **kwargs)


# ===================================================== #
# ===================================================== #
# ===================================================== #
# ===================================================== #
# ===================================================== #
def ax_plot_theta(ax, x, param_star, param_name, color="k", ylogscale=True):

    if param_star is not None:
        ax.plot(x)
        ax.axhline(param_star, linestyle="--", label=param_name, color=color)
    else:
        ax.plot(x, label=param_name)

    if ylogscale and (x[jnp.logical_not(jnp.isnan(x))] > 0).all():
        ax.set_yscale("log")

    ax.legend(loc="center left")
    return ax


@default_figure
def _plot_theta(
    x,
    id_to_plot=None,
    params_star=None,
    params_names=None,
    log_scale=True,
    fig=None,
):
    if id_to_plot is not None:
        x = x[id_to_plot,]

        if params_star is not None:
            params_star = params_star[id_to_plot,]
        if params_names is not None:
            params_names = params_names[id_to_plot]
    ntheta = x.shape[0]

    if params_names is None:
        params_names = [str(i) for i in range(ntheta)]

    if params_star is not None:
        params_star = jnp.hstack(params_star)
    else:
        params_names = [
            [n] + ["" for _ in range(x.shape[-1] - 1)] for n in params_names
        ]  # duplicate
        params_star = [None for i in range(ntheta)]

    for i in range(ntheta):
        ax = fig.add_subplot(ntheta, 1, i + 1)
        ax_plot_theta(
            ax,
            x[i],
            params_star[i],
            params_names[i],
            color=f"C{i}",
            ylogscale=log_scale,
        )

    if len(fig.axes) > 0:
        fig.axes[0].set_title("Parameter")

    return fig


@default_figure
def plot_theta(
    multi_estim,
    dim_ld=None,
    params_star=None,
    params_names=None,
    log_scale=True,
    id_to_plot=None,
    fig=None,
):
    multi_theta = get_all_theta(multi_estim)
    multi_theta = multi_theta[:dim_ld]
    return _plot_theta(
        multi_theta,
        id_to_plot=id_to_plot,
        params_star=params_star,
        params_names=params_names,
        log_scale=log_scale,
        fig=fig,
    )


@default_figure
def plot_theta_hd(
    multi_estim, dim_ld, params_star=None, params_names=None, log_scale=True, fig=None
):
    multi_theta = get_all_theta(multi_estim)
    multi_theta = multi_theta[dim_ld:]

    if params_star is not None:
        params_star = jnp.hstack(params_star)[dim_ld:]
    if params_names is not None:
        params_names = params_names[dim_ld:]

    return _plot_theta(
        x=multi_theta,
        params_star=params_star,
        params_names=params_names,
        log_scale=log_scale,
        fig=fig,
    )


# =======================================================================#


def plot_axvline(ax, lbd_set, bic, i, color, msg=""):
    lbd = lbd_set[i]

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
        rf"$\lambda$ = {lbd_set[i]:.3e}",
        ha="center",
        va="center",
        rotation="vertical",
        backgroundcolor="white",
    )

    return ax


def _plot_bic(
    ax,
    lbd_set,
    bic,
    argmin_bic,
    colors=["k", "w", "b"],
    msg=" = min(BIC)",
):

    ax_bic = ax.twinx()
    ax_bic.plot(lbd_set, bic, color=colors[0], linewidth=5, linestyle="-")
    ax_bic.plot(lbd_set, bic, color=colors[1], linewidth=3, linestyle="-")
    ax_bic.plot(lbd_set, bic, color=colors[0], linewidth=2, linestyle="--", label="BIC")
    ax_bic.set(ylabel="BIC Score")

    # minimum value of bic
    plot_axvline(ax_bic, lbd_set, bic, i=argmin_bic, color=colors[2], msg=msg)


@add_figure
def plot_reg_path(ax, reg_res: RegularizationPathRes, dim_ld: int = 0, fig=None):
    if isinstance(reg_res, RegularizationPathRes):
        multi_theta_hd = [res[-1].last_theta[dim_ld:] for res in reg_res.multi_run]
        lbd_set = reg_res.lbd_set

        ax.set_title("Regularization path")
        ax.set_xlabel(r"Regularization penalty ($\lambda$)")
        ax.set_ylabel(r"HD Parameter ($\beta$)")

        ax.set_xscale("log")

        ax.plot(lbd_set, multi_theta_hd)

        return _plot_bic(
            ax,
            lbd_set=lbd_set,
            bic=reg_res.bic[-1],
            argmin_bic=reg_res.argmin_bic,
        )


# def plot_box_plot_hd(theta, dim_ld=0, params_star=None, threshold=0):

#     fig = figure()
#     ax = fig.add_subplot(1, 1, 1)

#     multi_theta = jnp.array([t[dim_ld:] for t in theta]).T
#     num_support = (multi_theta != 0).sum(axis=1)

#     id_support = np.array(
#         [i for i in range(len(num_support)) if num_support[i] >= threshold]
#     )
#     xticks = [i + 1 for i in range(len(id_support))]

#     ax.boxplot(multi_theta[:, id_support])

#     if params_star is not None:
#         params_star = jnp.hstack(params_star)[dim_ld:]
#         ax.plot(xticks, params_star[id_support], "bs", label="true value")
#         ax.set_xticks(xticks, id_support)

#         ax.legend()

#     return fig, ax


def plot_mcmc(x, id_max=None):
    """plot an MCMC_chain"""

    if isinstance(x, dict):
        for mcmc in x.values():
            plot_mcmc(mcmc)
        return None

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


def get_dataframe_results(x, x_star, rows_names):
    x = jnp.array(x)

    rmse = jnp.sqrt(((x - x_star) ** 2).mean(axis=0))
    df = jnp.array(
        [x_star, x.mean(axis=0), x.var(axis=0), rmse, rmse / jnp.abs(x_star) * 100]
    ).T
    id_non_zero = jnp.logical_or(df[:, 1] != 0, df[:, 1] != df[:, 2])
    return pd.DataFrame(
        df[id_non_zero, :],
        columns=["real value", "mean", "variance", "RMSE", "RRMSE"],
        index=rows_names[id_non_zero],
    )


@default_figure
def plot_2_panel_selected_theta_hd(
    results, p_star, scenarios_labels, theta_name="$\\beta$", dim_ld: int = 0, fig=None
):
    from matplotlib.gridspec import GridSpec

    G = GridSpec(len(results), 3)

    params_star_hd = p_star[dim_ld:]
    theta = results.last_theta[:, :, -1, dim_ld:]
    non_zero = 3

    xticks = jnp.arange(0, theta.shape[-1]) + 1
    for i in range(len(results)):
        ax = plt.subplot(G[i, 0])

        myBoxplot(
            ax=ax,
            x=theta[i][:, :non_zero].T,
            xlabels=[f"{k+1}" for k in range(non_zero)],
        )
        ax.plot(
            xticks[:non_zero] - 1, params_star_hd[:non_zero], "bs", label="true value"
        )

        ax.legend()
        ax.set_ylabel(scenarios_labels[i])
        # == == == == #
        ax = plt.subplot(G[i, 1:])  # , sharey=ax)
        tt = theta[i][:, non_zero:].T
        theta_nonan = tt[jnp.array([~jnp.isnan(xx).any() for xx in tt]), :]

        # sdgplt.myBoxplot(ax = ax, x = theta_nonan)
        points = sum(
            [
                [(i + non_zero, xx) for xx in theta_nonan[i] if xx != 0]
                for i in range(theta_nonan.shape[0])
                if jnp.abs(theta_nonan[i]).sum() != 0
            ],
            [],
        )

        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            facecolors="none",
            edgecolors="k",
        )
        ax.hlines(0, xmin=non_zero, xmax=theta_nonan.shape[0], colors="k")

        xticks_nonzero = [non_zero + 1] + [
            (x + 1) * 25 for x in range((theta_nonan.shape[0] + non_zero) // 25)
        ]

        ax.set_xticks(xticks_nonzero, [str(x) for x in xticks_nonzero])

    ax = plt.subplot(G[0, 0])
    ax.set_title(
        f"Estimation of the {non_zero} non-zero components of {theta_name}", fontsize=15
    )
    ax = plt.subplot(G[0, 1:])  # , sharey=ax)
    ax.set_title(
        f"Estimation of the remaining zero components of {theta_name}", fontsize=15
    )

    return fig
