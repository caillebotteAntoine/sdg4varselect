"""
Module defining functions for generating plots.

This module provides utilities for plotting results from stochastic gradient descent
and MCMC sampling processes.

Author:
    Antoine Caillebotte (antoine.caillebotte@inrae.fr)
"""

from typing import Union
import matplotlib.pyplot as plt

import numpy as np

# import jax.numpy as jnp

from sdg4varselect.algo.sto_grad_descent_precond import (
    StochasticGradientDescentPrecond as SGD,
)
from sdg4varselect._mcmc import MCMC
from sdg4varselect.outputs import (
    GDResults,
    SGDResults,
    MultiGDResults,
    RegularizationPath,
    MultiRegularizationPath,
)

FIGSIZE = 8


def figure(height=None, width=None):
    """Creates a matplotlib figure with specified height and width.

    Parameters
    ----------
    height : float, optional
        The height of the figure. If None, a default value of FIGSIZE is used.
    width : float, optional
        The width of the figure. If None, a default value of FIGSIZE is used.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure with specified dimensions.
    """
    fig = plt.figure()
    fig.set_figheight(FIGSIZE if height is None else height)
    fig.set_figwidth(FIGSIZE if width is None else width)
    return fig


def ax(height=None, width=None):
    """Creates a matplotlib Axes object from an figure with specified height and width.

    Parameters
    ----------
    height : float, optional
        The height of the axes figure. If None, a default value of FIGSIZE is used.
    width : float, optional
        The width of the axes figure. If None, a default value of FIGSIZE is used.

    Returns
    -------
    matplotlib.axes._axes.Axes
        A subplot axes within the created figure.
    """
    fig = figure(height=height, width=width)
    return fig.add_subplot(1, 1, 1)


# ===================================================== #
# ===================================================== #
def plot_mcmc(x: Union[SGD, MCMC, dict], id_max=None):
    """Plots data from an MCMC chain or Stochastic Gradient Descent (SGD) latent variables.

    Parameters
    ----------
    x : SGD, MCMC, or dict
        The object containing MCMC chain or SGD latent variables to plot.
        If `x` is a dictionary, each value will be plotted recursively.
    id_max : int, optional
        The maximum number of iterations to plot. If None, all available iterations are plotted.

    Returns
    -------
    matplotlib.figure.Figure or list of matplotlib.figure.Figure
        The figure or list of figures created for the MCMC chain plot.
    """
    if isinstance(x, SGD):
        return plot_mcmc(x.latent_variables, id_max=id_max)
    if isinstance(x, dict):
        return [plot_mcmc(mcmc, id_max=id_max) for mcmc in x.values()]

    assert isinstance(x, MCMC)
    return x.plot(fig=figure(), id_max=id_max)


def plot_theta(
    x,
    *,
    fig=None,
    params_names: list = None,
    id_to_plot: Union[list[list[int]], list[int]] = None,
    log_scale: bool = True,
):  # pylint: disable=too-many-arguments
    """Plots model parameters from stochastic gradient descent results.

    Parameters
    ----------
    x : MultiGDResults, GDResults or SGDResults
        Results of stochastic gradient descent.
    fig : matplotlib.figure.Figure or list of matplotlib.figure.Figure, optional
        Figure(s) to use for plotting. If None, a new figure is created.
    params_names : list of str, optional
        Names of the parameters to label in the plot.
    id_to_plot : list of int or list of list of int, optional
        Indices of parameters to plot. If None, all parameters are plotted.
    log_scale : bool, default=True
        Whether to use logarithmic scale for plotting.

    Returns
    -------
    list of matplotlib.figure.Figure or matplotlib.figure.Figure
        The figure(s) created for the parameter plot.
    """
    if isinstance(x, (GDResults, SGDResults)):
        x = MultiGDResults(results=[x])

    assert isinstance(x, MultiGDResults)

    if params_names is not None:
        params_names = np.array(params_names)

    if id_to_plot is not None:
        if not isinstance(id_to_plot[0], list):
            id_to_plot = [id_to_plot]

        if fig is None:
            fig = [figure() for _ in id_to_plot]

        if isinstance(fig, (list, np.ndarray)):
            assert len(id_to_plot) == len(fig)

            return [
                x.shrink(col=col).plot_theta(
                    fig[i],
                    params_names[np.array(col)] if params_names is not None else None,
                    log_scale,
                )
                for i, col in enumerate(id_to_plot)
            ]

    if fig is None:
        fig = figure()
    return x.plot_theta(fig, params_names, log_scale)


def plot_regpath(x, *, fig=None, P=None):
    """Plots the regularization path, showing parameter values and BIC/eBIC scores.

    This method visualizes the evolution of parameter values across `lambda` values,
    as well as the BIC and eBIC scores. For each score, a vertical line is drawn at the
    optimal `lambda` where the score is minimized.

    Parameters
    ----------
    x : RegularizationPath
        Results of the regularizationPath.
    fig : matplotlib.figure.Figure or list of matplotlib.figure.Figure, optional
        Figure(s) to use for plotting. If None, a new figure is created.
    P : int
        Number of parameters to display in the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the parameter plots.
    """
    assert isinstance(x, RegularizationPath)

    if fig is None:
        fig = figure()

    return x.plot(fig, P=P)


def plot_estimation(
    x,
    *,
    fig=None,
    params_names: list = None,
    id_to_plot: Union[list[list[int]], list[int]] = None,
    xlabels=None,
    nrows=4,
    ncols=4,
    **kwargs,
):  # pylint: disable=too-many-arguments
    """
    Plot estimation results for a given set of MultiRegularizationPath objects.

    This function generates subplots to visualize the parameter estimation results.
    It supports multiple `MultiRegularizationPath` objects, plotting them on a grid
    defined by `nrows` and `ncols`, and highlights specific components based on their
    indices.

    Parameters
    ----------
    x : list of MultiRegularizationPath
        List of MultiRegularizationPath objects to be plotted.
    fig : matplotlib.figure.Figure, optional
        Matplotlib figure to add subplots to If None, a new figure is created.
    params_names : list, optional
          List of parameter names corresponding to the components being plotted. If None, no titles are shown.
    id_to_plot : list of int or list of list of int, optional):
        Indices of the components  to be plotted. If None, all components are plotted.
    xlabels : list, optional
        List of labels for the x-axis ticks. Default is None.
    nrows : int, optional
        Number of rows in the subplot grid. Default is 4.
    ncols : int, optional
        Number of columns in the subplot grid. Default is 4.
    **kwargs:
        Additional arguments passed to `boxplot`.

    Returns
    -------
    list of matplotlib.figure.Figure
        The Matplotlib figure object containing the generated subplots.

    Raises
    ------
    AssertionError:
        If any input validation fails (e.g., invalid index, shape mismatch).
    """

    if isinstance(x, MultiRegularizationPath):
        x = [x]
    assert isinstance(x, list)
    for xx in x:
        assert isinstance(xx, MultiRegularizationPath)

    p_star = x[0].theta_star
    x = np.array([xx.last_theta for xx in x]).T

    if id_to_plot is None:
        id_to_plot = list(range(x.shape[0]))

    for ii in id_to_plot:
        assert 0 <= ii < x.shape[0]

    assert len(id_to_plot) <= nrows * ncols

    if fig is None:
        fig = figure()

    subfigs = fig.subfigures(nrows, ncols)

    def boxplot(x, *, fig, hline=None, facecolor="W", title=None):
        axes = fig.add_subplot(1, 1, 1)
        bp = axes.boxplot(
            x,
            patch_artist=True,
            labels=xlabels if xlabels is not None else None,
            **kwargs,
        )

        for patch in bp["boxes"]:
            patch.set(facecolor=facecolor)

        for median in bp["medians"]:
            median.set_color("black")

        if hline is not None:
            axes.axhline(y=hline, color="k")

        if title is not None:
            axes.set_title(title)

    for i, col in enumerate(id_to_plot):
        boxplot(
            x=x[col],
            fig=subfigs[i // ncols][i % ncols],
            hline=None if p_star is None else p_star[col],
            facecolor=f"C{i}",
            title=None if params_names is None else params_names[col],
            **kwargs,
        )

    return fig


def plot_selection(
    x, id_non_zeros, id_zeros, *, fig=None, params_names=None, width_ratios=(1, 4)
):  # pylint: disable=too-many-arguments
    """
    Plot selection of non-zero and zero components for a given set of results.

    This function generates a figure with two subplots: one for the non-zero components
    and another for the zero components. It can handle both individual `MultiRegularizationPath`
    and `MultiGDResults` objects or lists of such objects. Non-zero components are visualized
    with boxplots, and zero components are visualized with scatter plots.

    Parameters
    ----------
    x : MultiRegularizationPath or MultiGDResults or list
        A `MultiRegularizationPath` or `MultiGDResults` object, or a list of such objects to be visualized.
    id_non_zeros : list[int]
        Indices of the non-zero components to be visualized.
    id_zeros : list[int]
        Indices of the zero components to be visualized.
    fig : matplotlib.figure.Figure, optional
          Matplotlib figure to add subplots to. If None, a new figure is created.
    params_names : list, optional
          List of parameter names for the components being visualized. If None, no titles are shown.
    width_ratios : tuple, optional
          Width ratio for the two subfigures (non-zero and zero components). Default is

    Returns
    -------
    list of matplotlib.figure.Figure
        list of figures created

    Raises
    ------
    AssertionError:
        If any input validation fails (e.g., invalid index, shape mismatch).
    """
    id_zeros = np.array(id_zeros)
    id_non_zeros = np.array(id_non_zeros)
    params_names = np.array(params_names)

    if isinstance(x, list):
        assert any(
            isinstance(xx, (MultiGDResults, MultiRegularizationPath)) for xx in x
        )
        if fig is not None and isinstance(fig, list):
            assert len(x) == len(fig)

        return [
            plot_selection(
                xx,
                fig=None if fig is None else fig[i],
                params_names=params_names,
                id_zeros=id_zeros,
                id_non_zeros=id_non_zeros,
            )
            for i, xx in enumerate(x)
        ]

    p_star = x.theta_star
    x = x.last_theta.T

    if fig is None:
        fig = figure()

    fig_non_zeros, fig_zeros = fig.subfigures(1, 2, width_ratios=width_ratios)

    # === NON ZEROS === #
    axes = fig_non_zeros.subplots(1, 1)
    axes.set_title("Estimation of the non-zero components", fontsize=15)

    axes.boxplot(
        x=x[id_non_zeros].T,
        labels=None if params_names is None else params_names[id_non_zeros],
    )
    if p_star is not None:
        xticks = 1 + np.arange(0, len(id_non_zeros))
        axes.plot(xticks, p_star[id_non_zeros], "bs", label="true value")
        axes.legend()

    # === ZEROS === #
    axes = fig_zeros.subplots(1, 1)
    axes.set_title("Estimation of the remaining zero components", fontsize=15)

    tt = x[id_zeros]
    points = sum(
        [
            [(i, xx) for xx in tt[i] if xx != 0]
            for i in range(tt.shape[0])
            if np.abs(tt[i]).sum() != 0
        ],
        [],
    )

    ticks = np.array([0, len(id_zeros) - 1])

    if len(points) != 0:
        points = np.array(points)
        axes.scatter(points[:, 0], points[:, 1], facecolors="none", edgecolors="k")

        ticks = np.unique(
            np.array(
                [0] + list(np.unique(points[:, 0])) + [len(id_zeros) - 1],
                dtype=np.int64,
            )
        )

    axes.set_xticks(ticks=ticks)
    if params_names is not None:
        axes.set_xticklabels(labels=params_names[id_zeros[ticks]])

    axes.hlines(0, xmin=0, xmax=tt.shape[0] - 1, colors="k")
    return [fig_non_zeros, fig_zeros]
