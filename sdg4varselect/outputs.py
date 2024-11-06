"""
Module for handling results in the sdg4varselect package.

This module defines classes and functions to load and save results related to
model training and evaluation.

Created by antoine.caillebotte@inrae.fr
"""

import dataclasses

import gzip
import pickle

from copy import deepcopy
from typing import Type, Generator

from datetime import timedelta
import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.exceptions import Sdg4vsNanError

###########################################################################################################


def _get_filename(
    model: type[AbstractModel],
    root: str = "",
    filename_add_on: str = "",
):
    """Generate a filename for saving or loading model results.

    Parameters
    ----------
    model : type[AbstractModel]
        The model class used to generate the filename.
    root : str, optional
        The root directory for the filename (default is "").
    filename_add_on : str, optional
        An optional suffix to add to the filename (default is "").

    Returns
    -------
    str
        The generated filename.
    """
    filename = model.name

    if filename_add_on != "":
        filename = filename + "_" + filename_add_on

    if root != "":
        filename = root + "/" + filename

    return filename


class Sdg4vsResults:
    """Class to handle results for the sdg4varselect package."""

    @staticmethod
    def load(
        model: type[AbstractModel],
        root: str = "",
        filename_add_on: str = "",
    ) -> "Sdg4vsResults":
        """Load results from a file.

        Parameters
        ----------
        model : type[AbstractModel]
            The model class used to determine the filename.
        root : str, optional
            The root directory for the filename (default is "").
        filename_add_on : str, optional
            An optional suffix to add to the filename (default is "").

        Returns
        -------
        Sdg4vsResults
            The loaded results object.
        """
        filename = _get_filename(model, root, filename_add_on)
        out = pickle.load(gzip.open(f"{filename}.pkl.gz", "rb"))
        print(f"{filename} LOADED !")
        return out

    def save(
        self,
        model: type[AbstractModel],
        root: str = "",
        filename_add_on: str = "",
    ):
        """Save the results object to a file.

        Parameters
        ----------
        model : type[AbstractModel]
            The model class used to determine the filename.
        root : str, optional
            The root directory for the filename (default is "").
        filename_add_on : str, optional
            An optional suffix to add to the filename (default is "").
        """
        filename = _get_filename(model, root, filename_add_on)
        pickle.dump(self, gzip.open(f"{filename}.pkl.gz", "wb"))
        print(f"{filename} SAVED !")


###########################################################################################################


@dataclasses.dataclass
class GDResults(Sdg4vsResults):
    """Class to handle results from gradient descent optimization.

    Attributes
    ----------
    theta : jnp.ndarray
        The estimated model parameters iterations after gradient descent.
    theta_reals1d : jnp.ndarray, optional
        Real-valued 1D parameters, for internal model transformations.
    fim : jnp.ndarray, optional
        Fisher Information Matrix for parameter estimation precision.
    grad : jnp.ndarray, optional
        The gradient values calculated during optimization.
    chrono : timedelta, default=timedelta()
        The duration of the gradient descent process.
    log_likelihood : jnp.ndarray, optional
        The log-likelihood value associated with the optimization process.
    """

    theta: jnp.ndarray
    theta_reals1d: jnp.ndarray = None
    fim: jnp.ndarray = None
    grad: jnp.ndarray = None
    chrono: timedelta = timedelta()
    log_likelihood: jnp.ndarray = jnp.nan

    @classmethod
    def new_from_list(cls, sdg_res, chrono) -> "GDResults":
        """Create a new GDResults object from a list of results.

        Parameters
        ----------
        sdg_res : list
            A list containing the gradient descent results.
        chrono : timedelta
            The duration of the gradient descent process.

        Returns
        -------
        GDResults
            An instance of GDResults define with the provided data.
        """
        res = [
            [sdg_res[i][j] for i in range(len(sdg_res))] for j in range(len(sdg_res[0]))
        ]

        return cls(
            theta=jnp.array(res[0]),
            theta_reals1d=None,
            fim=res[2],
            grad=jnp.array(res[1]),
            chrono=chrono,
            log_likelihood=jnp.nan,
        )

    def shrink(self, row=None, col=None):
        """Reduce the dimensions of theta and grad based on provided indices.

        Parameters
        ----------
        row : int, optional
            Row indices to keep.
        col : int, optional
            Column indices to keep.

        Returns
        -------
        GDResults
            A new GDResults instance with reduced dimensions.
        """
        assert row is not None or col is not None, "row or col must be provided !"

        out = deepcopy(self)
        if row is not None:
            out.theta = out.theta[row]
            out.grad = out.grad[row]
        if col is not None:
            out.theta = out.theta[:, col]

        return out

    def reshape(self, row, col=None):
        """Pad theta and grad attributes to match specified dimensions.

        Parameters
        ----------
        row : int
            Target number of rows for padding.
        col : int, optional
            Target number of columns for padding.

        Returns
        -------
        GDResults
            A new GDResults instance with reshaped theta and grad.
        """
        out = deepcopy(self)
        out.theta = jnp.pad(
            self.theta,
            (
                (0, row - self.theta.shape[0]),
                (0, 0 if col is None else (col - self.theta.shape[1])),
            ),
            constant_values=jnp.nan,
        )

        out.grad = jnp.pad(
            self.grad,
            ((0, row - self.grad.shape[0]), (0, 0)),
            constant_values=jnp.nan,
        )
        return out

    @property
    def last_theta(self) -> jnp.ndarray:
        """Get the last non-NaN row in theta.

        Returns
        -------
        jnp.ndarray
            The last non-NaN theta row.
        """
        id_not_all_nan = jnp.logical_not(jnp.isnan(self.theta).all(axis=1))
        out = self.theta[id_not_all_nan][-1]
        return out

    def reals1d_to_hstack_params(self, model) -> None:
        """Convert the theta reals 1-D parameters to a stacked parameter representation.

        Parameters
        ----------
        model : AbstractModel
            The model used to convert real-valued parameters.
        """
        tmp = [model.parametrization.reals1d_to_params(t) for t in self.theta]

        self.theta_reals1d = self.theta
        self.theta = jnp.array([model.hstack_params(t) for t in tmp])

    def make_it_lighter(self) -> None:
        """Reduce memory usage by keeping only first and last theta and grad values."""
        id_not_nan = jnp.logical_not(jnp.isnan(self.theta).any(axis=1))
        theta = self.theta[id_not_nan]
        grad = self.grad[id_not_nan]

        self.theta = jnp.array([theta[0], theta[-1]])
        self.fim = None
        self.grad = jnp.array([grad[0], grad[-1]])


###########################################################################################################


class SGDResults(GDResults):
    """
    Class to handle stochastic gradient descent (SGD) results.

    Attributes
    ----------
    latent_variables : dict[str, jnp.ndarray], optional
        A dictionary of latent variables tracked during SGD.
    """

    latent_variables: dict[str, jnp.ndarray] = None


###########################################################################################################


@dataclasses.dataclass
class MultiGDResults:
    """Iterable container for multiple GDResults instances.

    Attributes
    ----------
    results : list[Type[GDResults]]
        A list of GDResults instances.
    chrono : timedelta, default=timedelta()
        The cumulative duration of all GDResults processes.
    """

    results: list[Type[GDResults]] = dataclasses.field(default_factory=list)
    chrono: timedelta = timedelta()

    def __post_init__(self):
        """Initialize MultiGDResults by cleaning NaN errors and reshaping."""
        while Sdg4vsNanError in self.results:
            self.results.remove(Sdg4vsNanError)

        if len(self.results) > 0:
            max_row = max(run.theta.shape[-2] for run in self.results)
            self.reshape(max_row, col=None)

        for item in self.results:
            self.chrono += item.chrono

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i) -> Type[GDResults]:
        return self.results[i]

    def __iter__(self) -> Generator[None, Type[GDResults], None]:
        yield from self.results

    # === property === #
    @property
    def likelihood(self):
        """Get likelihood values from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of likelihood values.
        """
        return jnp.array([x.likelihood for x in self])

    @property
    def last_theta(self):
        """Get the last non-NaN row in theta from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of last non-NaN theta row.
        """
        return jnp.array([x.last_theta for x in self])

    @property
    def theta(self):
        """Get theta values from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of theta values.
        """
        return jnp.array([x.theta for x in self])

    # ========================================= #
    def make_it_lighter(self):
        """Reduce memory usage in all contained GDResults instances."""
        for res in self:
            res.make_it_lighter()

    def reshape(self, row, col=None):
        """Pad theta and grad in all GDResults instances to match specified dimensions.

        Parameters
        ----------
        row : int
            Target number of rows.
        col : int, optional
            Target number of columns.

        Returns
        -------
        MultiGDResults
            A new MultiGDResults instance with reshaped theta and grad.
        """
        out = deepcopy(self)
        for i, run in enumerate(out.results):
            out.results[i] = run.reshape(row, col)
        return out

    def shrink(self, row=None, col=None):
        """Reduce dimensions in all GDResults instances based on provided indices.

        Parameters
        ----------
        row : int, optional
            Row indices to keep.
        col : int, optional
            Column indices to keep.

        Returns
        -------
        MultiGDResults
            A new MultiGDResults instance with theta and grad with reduced dimensions.
        """
        out = deepcopy(self)
        for i, run in enumerate(out.results):
            out.results[i] = run.shrink(row, col)
        return out

    def sort(self):
        """Sort results based on likelihood values."""
        self.results = sorted(
            self.results,
            key=lambda x: (
                -x.likelihood if len(x.likelihood.shape) == 0 else -x.likelihood[-1]
            ),
        )

    # ===================================================== #
    # ===================================================== #

    def plot_theta(
        self,
        fig=None,
        params_star: jnp.ndarray = None,
        params_names: list[str] = None,
        log_scale: bool = True,
    ):
        """Plot model parameters across all results.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure used for plotting.
        params_star : array-like, optional
            Reference parameter values.
        params_names : list of str, optional
            Names of parameters for labeling.
        log_scale : bool, default=True
            Whether to use a logarithmic scale on the y-axis.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the parameter plots.

        Notes
        -----
        If `log_scale` is set to True, the y-axis will be set to a logarithmic scale
        if all non-NaN values in `t` are positive.
        """

        def ax_plot_theta(t: jnp.ndarray, param_star, param_name=None, color="k"):
            """Plot a parameter's trajectory and optionally add a reference line.

            Parameters
            ----------
            t : jnp.ndarray
                Array of parameter values to plot over iterations.
            param_star : float or None
                The reference (target) parameter value to plot as a horizontal line.
                If None, no reference line is plotted.
            param_name : str, optional
                Name of the parameter, used as the label if param_star is provided.
                Default is None.
            color : str, optional
                Color for the reference line if param_star is provided. Default is "k" (black).
            """

            ax.plot(t)
            if param_star is not None:
                ax.axhline(param_star, linestyle="--", label=param_name, color=color)
                if param_name is not None:
                    ax.legend(loc="center left")
            else:
                ax.set_title(param_name)

            if log_scale and (t[jnp.logical_not(jnp.isnan(t))] > 0).all():
                ax.set_yscale("log")

        x = self.theta.T
        if params_star is not None:
            assert x.shape[0] == params_star.shape[0]

        if params_names is not None:
            # params_names = np.array(params_names)
            assert x.shape[0] == params_names.shape[0]

        ntheta = x.shape[0]

        for i in range(ntheta):
            ax = fig.add_subplot(ntheta, 1, i + 1)
            ax_plot_theta(
                x[i],
                None if params_star is None else params_star[i],
                None if params_names is None else params_names[i],
                color=f"C{i}",
            )

        if len(fig.axes) > 0:
            fig.axes[0].set_title("Parameter")

        return fig
