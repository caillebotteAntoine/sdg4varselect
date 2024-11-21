"""
Module for handling results in the sdg4varselect package.

This module defines classes and functions to load and save results related to
model training and evaluation.

Created by antoine.caillebotte@inrae.fr
"""

import os
import dataclasses

import gzip
import pickle

from copy import deepcopy
from typing import Type, Generator

from datetime import timedelta
import jax.numpy as jnp


from sdg4varselect._criterion_bic_ebic import BIC, eBIC
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


@dataclasses.dataclass(kw_only=True)
class Sdg4vsResults:
    """Class to handle results for the sdg4varselect package.

    Attributes
    ----------
    chrono : timedelta, default=timedelta()
        The duration of the gradient descent process.
    """

    chrono: timedelta = timedelta()

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


@dataclasses.dataclass(kw_only=True)
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

    theta: jnp.ndarray = None
    theta_reals1d: jnp.ndarray = None
    fim: jnp.ndarray = None
    grad: jnp.ndarray = None
    log_likelihood: jnp.ndarray = jnp.nan
    bic: jnp.ndarray = None
    ebic: jnp.ndarray = None

    # def expand(self, support):
    #     def _expand(x):
    #         shape = (x.shape[0], support.shape[0])
    #         x_expand = jnp.zeros(shape=shape)
    #         return = x_expand.at[:, jnp.where(support)[0]].set(x)

    #     row, _ = self.theta.shape
    #     theta_expand = jnp.zeros(shape=(row, support.shape[0]))
    #     self.theta = theta_expand.at[:, jnp.where(support)[0]].set(self.theta)

    # def __add__(self, x):
    #     return GDResults(
    #         theta=jnp.concatenate([self.theta, x.theta]),
    #         theta_reals1d=jnp.concatenate([self.theta_reals1d, x.theta_reals1d]),
    #         grad=jnp.concatenate([self.grad, x.grad]),
    #         fim=self.fim + x.fim,
    #         chrono=self.chrono + x.chrono,
    #     )

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
        )

    def update_bic(self, model):
        """Update bic and ebic values.

        Parameters
        ----------
        model : object
            Model object containing the number of parameters (`P`) and the sample size (`N`)
            required for BIC and eBIC computation.
        """
        P = model.P
        N = model.N

        self.bic = BIC(self.last_theta[-P:], self.log_likelihood, N)
        self.ebic = eBIC(self.last_theta[-P:], self.log_likelihood, N)

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

    def pad(self, row, col=None):
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
        if out.theta is not None:
            out.theta = jnp.pad(
                out.theta,
                (
                    (0, row - out.theta.shape[0]),
                    (0, 0 if col is None else (col - out.theta.shape[1])),
                ),
                constant_values=jnp.nan,
            )

        if out.grad is not None:
            out.grad = jnp.pad(
                out.grad,
                ((0, row - out.grad.shape[0]), (0, 0)),
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
        self.fim = (self.fim[0], self.fim[1])
        self.grad = jnp.array([grad[0], grad[-1]])


###########################################################################################################


@dataclasses.dataclass(kw_only=True)
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


@dataclasses.dataclass(kw_only=True)
class MultiGDResults(Sdg4vsResults):
    """Iterable container for multiple GDResults instances.

    Attributes
    ----------
    results : list[Type[GDResults]]
        A list of GDResults instances.
    chrono : timedelta, default=timedelta()
        The cumulative duration of all GDResults processes.
    """

    results: list[Type[GDResults]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Initialize MultiGDResults by cleaning NaN errors and reshaping."""
        while Sdg4vsNanError in self.results:
            self.results.remove(Sdg4vsNanError)

        if len(self.results) > 0:
            max_row = max(run.theta.shape[-2] for run in self.results)
            self.pad(max_row, col=None)

        for item in self.results:
            self.chrono += item.chrono

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i: int) -> Type[GDResults]:
        return self.results[i]

    def __setitem__(self, i: int, res: Type[GDResults]) -> None:
        self.results[i] = res

    def __iter__(self) -> Generator[None, Type[GDResults], None]:
        yield from self.results

    # === property === #
    @property
    def log_likelihood(self):
        """Get log_likelihood values from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of log_likelihood values.
        """
        return jnp.array([x.log_likelihood for x in self])

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

    @property
    def last_theta_reals1d(self):
        """Get the last non-NaN row in theta_reals1d from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of last_reals1d non-NaN theta row.
        """
        return jnp.array([x.last_theta for x in self])

    @property
    def theta_reals1d(self):
        """Get theta_reals1d values from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of theta_reals1d values.
        """
        return jnp.array([x.theta_reals1d for x in self])

    @property
    def bic(self):
        """Get bic values from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of bic values.
        """
        return jnp.array([x.bic for x in self])

    @property
    def ebic(self):
        """Get ebic values from each GDResults instance.

        Returns
        -------
        jnp.ndarray
            Array of ebic values.
        """
        return jnp.array([x.ebic for x in self])

    # ========================================= #
    def update_bic(self, model):
        """Update the BIC and eBIC scores for the model based on the final estimated parameters.

        Parameters
        ----------
        model : object
            Model object containing the number of parameters (`P`) and the sample size (`N`)
            required for BIC and eBIC computation.
        """
        for res in self:
            res.update_bic(model)

    def make_it_lighter(self):
        """Reduce memory usage in all contained GDResults instances."""
        for res in self:
            res.make_it_lighter()

    def pad(self, row, col=None):
        """Pad theta and grad in all GDResults instances to match specified dimensions.

        Parameters
        ----------
        row : int
            Target number of rows.
        col : int, optional
            Target number of columns.
        """
        for i, run in enumerate(self.results):
            self.results[i] = run.pad(row, col)

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
        """Sort results based on log_likelihood values."""
        self.results = sorted(
            self.results,
            key=lambda x: (
                -x.log_likelihood
                if len(x.log_likelihood.shape) == 0
                else -x.log_likelihood[-1]
            ),
        )

    # ===================================================== #
    # ===================================================== #

    def plot_theta(
        self,
        fig,
        params_star: jnp.ndarray = None,
        params_names: list[str] = None,
        log_scale: bool = True,
    ):
        """Plot model parameters across all results.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
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


@dataclasses.dataclass(kw_only=True)
class RegularizationPath(MultiGDResults):
    """Class representing the regularization path of model parameters for varying regularization penalties.

    This class stores results from a multi-gradient descent (GD) algorithm across different values of
    a regularization parameter `lambda` (stored in `lbd_set`). It provides methods to standardize the
    computed regularization path and plot the results for model evaluation and selection.

    Parameters
    ----------
    lbd_set : jnp.ndarray
        Array of regularization parameter values (`lambda` values) used in the regularization path.

    Methods
    -------
    standardize()
        Standardizes the regularization path by selecting the model with the lowest Bayesian Information
        Criterion (BIC) or Extended Bayesian Information Criterion (eBIC) for each support.

    plot(fig, P)
        Plots the regularization path, showing the selected BIC and eBIC scores, as well as the parameter
        values across the range of `lambda` values.
    """

    lbd_set: jnp.ndarray = jnp.nan

    def __post_init__(self):
        super().__post_init__()

        assert len(self.lbd_set) == len(
            self.results
        ), f"lbd_set must have the same size of the results list: {len(self.lbd_set)} != {len(self.results)}"

    def standardize(self):
        """Standardizes the regularization path by selecting models with the lowest BIC/eBIC.

        This method iterates over the regularization path and, for each unique support of non-zero
        parameters, finds the model with the minimum BIC score. The BIC and eBIC values for that
        model are then assigned to all other models with the same support. This ensures consistency
        in evaluation metrics across models with identical support.

        Returns
        -------
        RegularizationPath
            A standardized version of the `RegularizationPath` object with consistent BIC/eBIC values
            for models with the same support.
        """
        out = deepcopy(self)
        k = 0

        while k < len(self):
            supp_k = self[k].last_theta != 0
            same_supp = []

            i = k
            supp_i = supp_k
            while i < len(self) and (supp_i == supp_k).all():
                same_supp.append(i)
                i += 1
                if i < len(self):
                    supp_i = self[i].last_theta != 0

            best_supp_id = k + self.ebic[jnp.array(same_supp)].argmin()
            k = i
            # print(same_supp, best_supp_id)
            for i in same_supp:
                out[i] = self[best_supp_id]
                out[i].chrono = self[i].chrono

        return out

    def plot(self, fig, P=None):
        """Plots the regularization path, showing parameter values and BIC/eBIC scores.

        This method visualizes the evolution of parameter values across `lambda` values,
        as well as the BIC and eBIC scores. For each score, a vertical line is drawn at the
        optimal `lambda` where the score is minimized.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object on which to plot.
        P : int
            Number of parameters to display in the plot.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the parameter plots.
        """

        def axvline_argmin(ax, x, color):
            lbd = self.lbd_set[jnp.argmin(x)]

            ax.axvline(x=lbd, color=color, linewidth=2, linestyle="--")
            ax.text(
                lbd,
                0.8 * x.max() + 0.2 * x.min(),
                rf"$\lambda$ = {lbd:.3e}",
                color=color,
                ha="center",
                va="center",
                rotation="vertical",
                backgroundcolor="white",
            )

        def plot_bic(
            ax, x, colors=("k", "w"), name=""
        ):  # pylint: disable=missing-return-doc, missing-return-type-doc

            (l1,) = ax.plot(
                self.lbd_set, x, color=colors[0], linewidth=5, linestyle="-", label=name
            )
            (l2,) = ax.plot(
                self.lbd_set, x, color=colors[1], linewidth=3, linestyle="--"
            )
            ax.set(ylabel="Score")

            axvline_argmin(ax, x, color=colors[0])
            return (l1, l2)

        ax = fig.subplots(1, 1)

        if P is None:
            P = 0

        multi_theta_hd = self.last_theta[:, -P:]
        ax.plot(self.lbd_set, multi_theta_hd)

        ax.set_title("Regularization path")
        ax.set_xlabel(r"Regularization penalty ($\lambda$)")
        ax.set_ylabel(r"Parameter")

        ax.set_xscale("log")

        ax_bic = ax.twinx()
        _ = plot_bic(ax_bic, self.bic, colors=["b", "w"], name="BIC")
        _ = plot_bic(ax_bic, self.ebic, colors=["r", "w"], name="eBIC")
        # fig.legend([lines_bic, lines_ebic], ["BIC", "eBIC"])
        return fig


@dataclasses.dataclass(kw_only=True)
class MultiRegularizationPath(Sdg4vsResults):
    """Iterable container for multiple RegularizationPath instances.

    Attributes
    ----------
    results : list[Type[RegularizationPath]]
        A list of RegularizationPath instances.
    """

    results: list[Type[RegularizationPath]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        for item in self.results:
            self.chrono += item.chrono

    @staticmethod
    def load(
        model: type[AbstractModel],
        root: str = "",
        filename_add_on: str = "",
        add_on_id: list[int] = None,
        clean_files=False,
    ) -> "MultiRegularizationPath":
        """Load results from a file.

        Parameters
        ----------
        model : type[AbstractModel]
            The model class used to determine the filename.
        root : str, optional
            The root directory for the filename (default is "").
        filename_add_on : str, optional
            An optional suffix to add to the filename (default is "").
        add_on_id : list[int], optional
            A list of integer IDs to append to the filename. If not empty, incremental IDs will
            be tested for loading. If files are found, they will be merged.
        clean_files : bool, optional
            If True, intermediate files will be deleted after merging (default is False).

        Returns
        -------
        MultiRegularizationPath
            An object containing the merged results if multiple files were found,
            or the results from a single file if no IDs were provided or only one file was found.

        Raises
        ------
        FileNotFoundError
            If no files are found for the specified IDs and no fallback results can be loaded.
        """

        if add_on_id is not None:
            assert isinstance(add_on_id, list)

            res = []
            try:
                for i in add_on_id:
                    res.append(
                        RegularizationPath.load(model, root, f"{filename_add_on}{i}")
                    )
                    add_on_id.append(i + 1)
            except FileNotFoundError as exc_subfile:
                add_on_id.pop()

                if len(res) != 0:
                    print(f"{len(res)} files found, merged into a single file :")
                    out = MultiRegularizationPath(results=res)
                    filename = f"{filename_add_on}all_{min(add_on_id)}_{max(add_on_id)}"
                    out.save(model=model, root=root, filename_add_on=filename)
                    if clean_files:
                        for i in add_on_id:
                            os.remove(
                                _get_filename(
                                    model, root, f"{filename_add_on}{i}.pkl.gz"
                                )
                            )

                else:
                    raise exc_subfile
        else:
            out = Sdg4vsResults.load(
                model=model, root=root, filename_add_on=filename_add_on
            )

        return out

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i: int) -> Type[RegularizationPath]:
        return self.results[i]

    def __setitem__(self, i: int, res: Type[RegularizationPath]) -> None:
        self.results[i] = res

    @property
    def last_theta(self):
        """Get the last non-NaN row in theta from each RegularizationPath instance where ebic have minimal.

        Returns
        -------
        jnp.ndarray
            Array of last non-NaN theta row.
        """
        return jnp.array([x[jnp.argmin(x.ebic)].last_theta for x in self.results])
