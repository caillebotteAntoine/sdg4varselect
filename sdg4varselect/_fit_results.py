"""
Module for handling outputs in the sdg4varselect package.

Created by antoine.caillebotte@inrae.fr
"""

import dataclasses

import gzip
import pickle

from copy import deepcopy

from datetime import timedelta
import jax.numpy as jnp

from sdg4varselect.models.abstract.abstract_model import AbstractModel

###########################################################################################################


def _get_filename(
    filename: str,
    root: str = "",
    filename_add_on: str = "",
):
    """Generate a filename for saving or loading model results.

    Parameters
    ----------
        filename : str
            filename for example the out put of the method name of an model object.
    root : str, optional
        The root directory for the filename (default is "").
    filename_add_on : str, optional
        An optional suffix to add to the filename (default is "").

    Returns
    -------
    str
        The generated filename.
    """
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
    chrono_iter : timedelta, default=timedelta()
        The duration of the each step of the gradient descent process.
    """

    chrono: timedelta = timedelta()
    chrono_iter: list[timedelta] = None

    @staticmethod
    def load(
        filename: str,
        root: str = "",
        filename_add_on: str = "",
    ) -> "Sdg4vsResults":
        """Load results from a file.

        Parameters
        ----------
        filename : str
            filename for example the out put of the method name of an model object.
        root : str, optional
            The root directory for the filename (default is "").
        filename_add_on : str, optional
            An optional suffix to add to the filename (default is "").

        Returns
        -------
        Sdg4vsResults
            The loaded results object.
        """
        fullfilename = _get_filename(filename, root, filename_add_on)
        out = pickle.load(gzip.open(f"{fullfilename}.pkl.gz", "rb"))
        print(f"{fullfilename} LOADED !")
        return out

    def save(
        self,
        filename: str,
        root: str = "",
        filename_add_on: str = "",
    ):
        """Save the results object to a file.

        Parameters
        ----------
        filename : str
            filename for example the out put of the method name of an model object.
        root : str, optional
            The root directory for the filename (default is "").
        filename_add_on : str, optional
            An optional suffix to add to the filename (default is "").
        """
        fullfilename = _get_filename(filename, root, filename_add_on)
        pickle.dump(self, gzip.open(f"{fullfilename}.pkl.gz", "wb"))
        print(f"{fullfilename} SAVED !")


###########################################################################################################


@dataclasses.dataclass(kw_only=True)
class FitResults(Sdg4vsResults):
    """Class to handle results from fitting algorithm.

    Attributes
    ----------
    theta : jnp.ndarray
        The estimated model parameters iterations after the fitting algorithm.
    theta_reals1d : jnp.ndarray
        Real-valued 1D parameters, for internal model transformations.
    theta_star :  jnp.ndarray
        The true model parameters used for data simulation.

    """

    theta: jnp.ndarray = None
    theta_reals1d: jnp.ndarray = None
    _theta_star: jnp.ndarray = None

    @property
    def theta_star(self):
        """Get true values of the parameter theta.

        Returns
        -------
        jnp.ndarray
            Array of theta_star values.
        """
        return self._theta_star

    @theta_star.setter
    def theta_star(self, x: jnp.ndarray):
        if not isinstance(x, jnp.ndarray):
            x = AbstractModel.hstack_params(x)
        assert len(x.shape) == 0 or (x.shape[-1] == self.theta.shape[-1])
        self._theta_star = x

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

    @property
    def last_theta_reals1d(self) -> jnp.ndarray:
        """Get the last non-NaN row in theta_reals1d.

        Returns
        -------
        jnp.ndarray
            The last non-NaN theta_reals1d row.
        """
        id_not_all_nan = jnp.logical_not(jnp.isnan(self.theta_reals1d).all(axis=1))
        out = self.theta_reals1d[id_not_all_nan][-1]
        return out

    def _shrink(self, row=None, col=None):
        """Reduce the dimensions of theta based on provided indices.

        Parameters
        ----------
        row : int, optional
            Row indices to keep.
        col : int, optional
            Column indices to keep.

        """
        assert row is not None or col is not None, "row or col must be provided !"

        if row is not None:
            row = jnp.array(row)
            self.theta = self.theta[row]
        if col is not None:
            col = jnp.array(col)
            self.theta = self.theta[:, col]

            if self.theta_star is not None:
                self.theta_star = self.theta_star[col]

    def shrink(self, row=None, col=None):
        """Reduce the dimensions of theta based on provided indices.

        Parameters
        ----------
        row : int, optional
            Row indices to keep.
        col : int, optional
            Column indices to keep.

        Returns
        -------
        FitResults
            A new FitResults instance with reduced dimensions.
        """
        out = deepcopy(self)
        out._shrink(row=row, col=col)  # pylint: disable=protected-access
        return out

    def _pad(self, row, col=None):
        """Pad theta attributes to match specified dimensions.

        Parameters
        ----------
        row : int
            Target number of rows for padding.
        col : int, optional
            Target number of columns for padding.

        """
        if self.theta is not None:
            self.theta = jnp.pad(
                self.theta,
                (
                    (0, row - self.theta.shape[0]),
                    (0, 0 if col is None else (col - self.theta.shape[1])),
                ),
                constant_values=jnp.nan,
            )

    def pad(self, row, col=None):
        """Pad theta attributes to match specified dimensions.

        Parameters
        ----------
        row : int
            Target number of rows for padding.
        col : int, optional
            Target number of columns for padding.

        Returns
        -------
        FitResults
            A new FitResults instance with reshaped theta
        """
        out = deepcopy(self)
        out._pad(self, row=row, col=col)  # pylint: disable=protected-access
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
        """Reduce memory usage by keeping only first and last theta values."""
        id_not_nan = jnp.logical_not(jnp.isnan(self.theta).any(axis=1))
        theta = self.theta[id_not_nan]
        self.theta = jnp.array([theta[0], theta[-1]])

        if self.theta_reals1d is not None:
            theta_reals1d = self.theta_reals1d[id_not_nan]
            self.theta_reals1d = jnp.array([theta_reals1d[0], theta_reals1d[-1]])
