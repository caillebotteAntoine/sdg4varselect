"""
Module for handling results in the sdg4varselect package.

This module defines classes and functions to load and save results related to
model training and evaluation.

Created by antoine.caillebotte@inrae.fr
"""

from dataclasses import dataclass

import gzip
import pickle

from datetime import timedelta
import jax.numpy as jnp

from sdg4varselect.models.abstract.abstract_model import AbstractModel

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


@dataclass
class GDResults(Sdg4vsResults):
    """Class to handle gradient descent results."""

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
            The time duration for the process.

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

    @property
    def last_theta(self) -> jnp.ndarray:
        """Return the last theta array from the theta attribute.

        Returns
        -------
        jnp.ndarray
            the last theta array from the theta attribute.

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
        """Reduce the memory usage of the results object by keeping only
        the first and last gradient descent parameters and gradients.
        """
        id_not_nan = jnp.logical_not(jnp.isnan(self.theta).any(axis=1))
        theta = self.theta[id_not_nan]
        grad = self.grad[id_not_nan]

        self.theta = jnp.array([theta[0], theta[-1]])
        self.fim = None
        self.grad = jnp.array([grad[0], grad[-1]])


###########################################################################################################


class SGDResults(GDResults):
    """Class to handle stochastic gradient descent results."""

    latent_variables: dict[str, jnp.ndarray] = None
