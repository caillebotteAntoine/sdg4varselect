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
)
from sdg4varselect.models.abstract.abstract_model import AbstractModel

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
    params_star: list = None,
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
    params_star : array-like, optional
        The true parameter values to be plotted as a baseline.
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
        x = MultiGDResults([x])

    assert isinstance(x, MultiGDResults)

    p_star = None
    if params_star is not None:
        p_star = AbstractModel.hstack_params(params_star)
    if params_star is not None:
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
                    p_star[np.array(col)] if p_star is not None else None,
                    params_names[np.array(col)] if params_names is not None else None,
                    log_scale,
                )
                for i, col in enumerate(id_to_plot)
            ]

    if fig is None:
        fig = figure()
    return x.plot_theta(fig, p_star, params_names, log_scale)


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
