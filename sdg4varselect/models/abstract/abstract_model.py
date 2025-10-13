"""
This module defines the `AbstractModel` class, which serves as a template for creating models.
"""

from abc import ABC, abstractmethod
import functools

from typing import Union

import jax.numpy as jnp
from jax import jit, jacfwd

import parametrization_cookbook.jax as pc

from sdg4varselect.exceptions import Sdg4vsException


def _check_initialization(fun):
    """decorator to check the correct initialization of a model
    Parameters:
    ----------
        fun: the function to be decorated

    Returns
    -------
        the decorated function
    """

    @functools.wraps(fun)
    def new_fun(
        self, *args, **kwargs
    ):  # pylint: disable=missing-return-doc, missing-return-type-doc
        if not self.is_initialized:
            raise Sdg4vsException(
                "The model has not been initiated and therefore cannot be used !"
            )
        return fun(self, *args, **kwargs)

    return new_fun


class AbstractModel(ABC):
    """AbstractModel: The base class for creating models.

    Subclasses must implement model-specific parametrization, likelihood, and sampling methods.

    Properties
    ----------
    is_initialized : bool
        Boolean flag indicating whether the model has been initialized.

    N : int
        The size parameter of the model.

    parametrization : Union[pc.Tuple, pc.NamedTuple]
        Returns the parametrization used in the model.

    name : str
        Abstract property returning the model name, based on model parameters.
    """

    def __init__(self, N: int, **kwargs):
        self._is_initialized: bool = False
        self._cst = kwargs
        self._n: int = N

        self._parametrization: pc.NamedTuple = None

        self.init()

    @abstractmethod
    def init_parametrization(self):
        """Initialize the model's parametrization.
        This method must set the _parametrization attribute.

        Raises
        ------
        NotImplementedError
            the method must be implemented in a subclass.
        """
        raise NotImplementedError("You must define parametrization in this function !")

    def init(self):
        """don't forget to call the mother init function at the end

        After calling this method is_initialized should be True and model should be ready for use
        """
        self.init_parametrization()
        self._is_initialized = True

    @property
    def is_initialized(self) -> bool:
        """Boolean flag indicating whether the model has been initialized.

        Returns:
            bool:
        """
        return self._is_initialized

    @property
    @_check_initialization
    def N(self) -> int:
        """The size parameter of the model

        Returns:
            int:
        """
        return self._n

    @property
    @abstractmethod
    @_check_initialization
    def name(self):
        """return a str called name, based on the parameter of the model
        Returns
        -------
        str
            the name of the model
        """
        raise NotImplementedError

    @property
    @_check_initialization
    def parametrization(self) -> Union[pc.Tuple, pc.NamedTuple]:
        """Returns the parametrization used in the model.

        Returns:
            Union[pc.Tuple, pc.NamedTuple]:
        """
        return self._parametrization

    @staticmethod
    def hstack_params(params) -> jnp.ndarray:
        """Return a copy of the params array collapsed into one dimension.

        Parameters
        ----------
        params : object
            the params array to be collapsed

        Returns
        -------
        jnp.ndarray
            A copy of the input array, flattened to one dimension.
        """
        return jnp.hstack([jnp.array(p).flatten("C") for p in list(params)])

    @_check_initialization
    def new_params(self, **kwargs) -> jnp.ndarray:
        """Initialize a parameter array

        Parameters
        ----------
        **kwargs : dict
            a dict containing the values of the parameters to be initialized

        Returns
        -------
        jnp.ndarray
            parameter array
        """
        theta_reals1d = self._parametrization.params_to_reals1d(**kwargs)
        return self._parametrization.reals1d_to_params(theta_reals1d)

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute likelihood array with individual components.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            parameters value used to the log-likelihood computation.
        **kwargs : dict
            a dict where all additional log_likelihood arguments can be found.

        Returns
        -------
        jnp.ndarray
            Array of log-likelihood components for each individual.
        """
        raise NotImplementedError

    @functools.partial(jit, static_argnums=0)
    def log_likelihood(self, theta_reals1d: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the log_likelihood for all the individuals

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            parameters value used to the log-likelihood computation.
        **kwargs : dict
            a dict where all additional log_likelihood arguments can be found.

        Returns
        -------
        jnp.ndarray
            log likelihood for all individuals
        """
        return self.log_likelihood_array(theta_reals1d, **kwargs).sum()

    @functools.partial(jit, static_argnums=0)
    def jac_log_likelihood(self, theta_reals1d: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Compute the Jacobian of the log_likelihood

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            parameters value used to the log-likelihood computation.
        **kwargs : dict
            a dict where all additional log_likelihood arguments can be found.

        Returns
        -------
        jnp.ndarray
            log likelihood as array of array each component for each individuals
        """

        return jacfwd(self.log_likelihood_array)(theta_reals1d, **kwargs)

    # ============================================================== #
    @abstractmethod
    @_check_initialization
    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Sample one data set for the model

        Parameters
        ----------
        params_star : object
            parameter used to sample the model
        prngkey : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample randomly the model
        **kwargs:
            additional data to be pass to any function used in sample

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
                - dict: Generated observations.
                - dict: Simulated latent variables.
        """
        raise NotImplementedError
