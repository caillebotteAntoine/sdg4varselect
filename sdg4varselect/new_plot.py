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


def ax(height=None, width=None):
    fig = figure(height=height, width=width)
    return fig.add_subplot(1, 1, 1)


# ===================================================== #
# ===================================================== #
# ===================================================== #
def ax_plot_theta(ax, x, param_star, param_name=None, color="k", ylogscale=True):

    if param_star is not None:
        ax.plot(x)
        ax.axhline(param_star, linestyle="--", label=param_name, color=color)
    else:
        ax.plot(x, label=param_name)

    if ylogscale and (x[jnp.logical_not(jnp.isnan(x))] > 0).all():
        ax.set_yscale("log")

    if param_name is not None:
        ax.legend(loc="center left")
    return ax


def plot_theta(
    x,
    params_star=None,
    params_names=None,
    id_to_plot=None,
    log_scale=True,
    fig=None,
):
    assert len(x.shape) == 3

    if id_to_plot is not None:
        if isinstance(fig, (list, np.ndarray)):
            assert len(id_to_plot) == len(fig)

            return [
                plot_theta(
                    x, params_star, params_names, id_to_plot[i], log_scale, fig[i]
                )
                for i in range(len(fig))
            ]

    return _plot_theta(x, params_star, params_names, id_to_plot, log_scale, fig)


def _plot_theta(
    x,
    params_star=None,
    params_names=None,
    id_to_plot=None,
    log_scale=True,
    fig=None,
):
    if id_to_plot is not None:
        id_to_plot = np.array(id_to_plot)

        if params_star is not None:
            assert x.shape[0] == params_star.shape[0]
            params_star = params_star[id_to_plot]
        if params_names is not None:
            params_names = np.array(params_names)
            assert x.shape[0] == params_names.shape[0]
            params_names = params_names[id_to_plot]

        x = x[id_to_plot,]

    ntheta = x.shape[0]

    for i in range(ntheta):
        ax = fig.add_subplot(ntheta, 1, i + 1)
        ax_plot_theta(
            ax,
            x[i],
            None if params_star is None else params_star[i],
            None if params_names is None else params_names[i],
            color=f"C{i}",
            ylogscale=log_scale,
        )

    if len(fig.axes) > 0:
        fig.axes[0].set_title("Parameter")

    return fig


# ===================================================== #
# ===================================================== #
# ===================================================== #
def boxplot(
    x,
    hline=None,
    hlabel=None,
    xlabels=None,
    id_to_plot=None,
    facecolor="w",
    title=None,
    fig=None,
    **kwargs,
):
    """plot estimation results using boxplot"""

    assert len(x.shape) <= 2
    assert xlabels is None or isinstance(xlabels, (str, list))

    if id_to_plot is not None:
        x = x[id_to_plot]
        if hline is not None:
            hline = hline[id_to_plot]

        if xlabels is not None:
            xlabels = np.array(xlabels)[id_to_plot]

    ax = fig.axes[0] if len(fig.axes) > 0 else fig.add_subplot(1, 1, 1)

    if len(x.shape) == 2:
        for i in range(x.shape[0]):
            boxplot(
                x=x[i],
                hline=None,
                xlabels=f"{i+1}" if xlabels is None else xlabels[i],
                facecolor=facecolor,
                fig=fig,
                title=title,
                positions=[i],
                **kwargs,
            )

        if hline is not None:
            ax.plot(hline, "bs", label="true value")
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

    if title is not None:
        ax.set_title(title)

    return ax


# ===================================================== #
# ===================================================== #
# ===================================================== #
def boxplot_estimation(x, id_to_plot, titles, xlabels, nrow=3, ncol=3, fig=None):
    subfigs = fig.subfigures(nrow, ncol)

    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j < len(id_to_plot):
                ii = id_to_plot[i * ncol + j]
                boxplot(
                    x=x[ii].T,
                    title=titles[ii],
                    xlabels=xlabels,
                    facecolor=f"C{ii}",
                    fig=subfigs[i][j],
                )

    return fig


# ===================================================== #
# ===================================================== #
# ===================================================== #
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


# ===================================================== #
# ===================================================== #
# ===================================================== #
def plot_axvline(ax, lbd_set, bic, i, color, msg=""):
    lbd = lbd_set[i]

    ax.axvline(
        x=lbd,
        color=color,
        linewidth=2,
        linestyle="--",
        # label=r"$\lambda$" + msg,
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
    ax_bic,
    lbd_set,
    bic,
    argmin_bic,
    colors=["k", "w", "b"],
    name="BIC",
):

    (l1,) = ax_bic.plot(
        lbd_set, bic, color=colors[0], linewidth=5, linestyle="-", label=name
    )
    (l2,) = ax_bic.plot(lbd_set, bic, color=colors[1], linewidth=3, linestyle="--")
    ax_bic.set(ylabel="Score")

    # minimum value of bic, msg=f" = min({name})"
    plot_axvline(ax_bic, lbd_set, bic, i=argmin_bic, color=colors[0])
    return (l1, l2)


def plot_reg_path(reg_res: RegularizationPathRes, D: int = 0, fig=None):
    if isinstance(reg_res, RegularizationPathRes):

        ax = fig.axes[0] if len(fig.axes) > 0 else fig.add_subplot(1, 1, 1)

        multi_theta_hd = [res[-1].last_theta[D:] for res in reg_res.multi_run]
        lbd_set = reg_res.lbd_set
        ax.plot(lbd_set, multi_theta_hd)

        ax.set_title("Regularization path")
        ax.set_xlabel(r"Regularization penalty ($\lambda$)")
        ax.set_ylabel(r"HD Parameter ($\beta$)")

        ax.set_xscale("log")

        ax_bic = ax.twinx()
        lines_bic = _plot_bic(
            ax_bic,
            lbd_set=lbd_set,
            bic=reg_res.bic[-1],
            argmin_bic=jnp.argmin(reg_res.bic[-1]),
            colors=["b", "w"],
            name="BIC",
        )

        lines_ebic = _plot_bic(
            ax_bic,
            lbd_set=lbd_set,
            bic=reg_res.ebic[-1],
            argmin_bic=jnp.argmin(reg_res.ebic[-1]),
            colors=["r", "w"],
            name="eBIC",
        )
        fig.legend([lines_bic, lines_ebic], ["BIC", "eBIC"])

        return fig


# ===================================================== #
# ===================================================== #
# ===================================================== #
