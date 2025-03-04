"""
Module for abstract class AbstractMixedEffectsModel.

Create by antoine.caillebotte@inrae.fr
"""

from abc import abstractmethod
import functools

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    AbstractLatentVariablesModel,
)

from sdg4varselect.exceptions import Sdg4vsWrongParametrization


class AbstractMixedEffectsModel(AbstractModel, AbstractLatentVariablesModel):
    """
    The most abstract mixed effects model that can be defined.

    This class provides an interface for mixed-effects models with latent variables.
    It combines functionality from both the AbstractModel and AbstractLatentVariablesModel
    base classes.

    Parameters
    ----------
    N : int
        Number of observations.
    J : int
        Number of time points or other repeated measures.
    me_name : list of str
        Names of the mixed effect variables.
    """

    def __init__(
        self,
        N: int,
        J: int,
        **kwargs,
    ):
        AbstractModel.__init__(self, N, **kwargs)
        AbstractLatentVariablesModel.__init__(self, size=N)

        self._j = J

    @property
    def J(self):
        """int: Number of time points or repeated measures."""
        return self._j

    def init(self):
        """Initialize the model by verifying required parameters.

        This method sets the model to a ready state by verifying that all
        mandatory parameters are present in the parametrization. After
        initialization, `is_initialized` should be True, and the model
        is ready for use.

        Raises
        ------
        Sdg4vsWrongParametrization
            If a mandatory parameter (cov_latent, var_residual) is missing.
        """
        AbstractModel.init(self)
        mandatory_parameter = ["cov_latent", "var_residual"]
        for mandatory_name in mandatory_parameter:
            if (
                mandatory_name not in self._parametrization.idx_params._fields
            ):  # _parms:
                raise Sdg4vsWrongParametrization(
                    f"parametrization must have {mandatory_name} defined !"
                )

        self._is_initialized = True

    # ============================================================== #
    @abstractmethod
    def mixed_effect_function(self, params, *args, **kwargs):
        """Define the nonlinear function for the mixed effects model.

        Parameters
        ----------
        params : dict
            Parameters of the model.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_without_prior(self, theta_reals1d, **kwargs) -> jnp.ndarray:
        """Compute the log-likelihood without Gaussian prior.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Parameters used to the log-likelihood computation.
        **kwargs : dict
            Additional keyword arguments used in the mixed_effect_function

        Returns
        -------
        jnp.ndarray
            Log-likelihood values for each individual without Gaussian prior.
        """
        params = self.parametrization.reals1d_to_params(theta_reals1d)

        Y = kwargs["Y"]
        mem_obs_time = kwargs["mem_obs_time"]

        N, J = Y.shape
        assert mem_obs_time.shape == (N, J)

        pred = self.mixed_effect_function(
            params, mem_obs_time, **self._cst, **kwargs
        )  # shape = (N,J)

        j_nan = J - jnp.isnan(Y).sum(axis=1)

        var_residual = params.var_residual
        log_likelihood_mem = -j_nan / 2 * jnp.log(
            2 * jnp.pi * var_residual
        ) - jnp.nansum((Y - pred) ** 2, axis=1) / (2 * var_residual)

        assert log_likelihood_mem.shape == (N,)
        return log_likelihood_mem

    @functools.partial(jit, static_argnums=0)
    def log_likelihood_only_prior(self, theta_reals1d, **kwargs) -> jnp.ndarray:
        """Compute log-likelihood with only the Gaussian prior.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Parameters used to the log-likelihood computation.
        **kwargs : dict
            a dict where all additional log_likelihood arguments can be found.

        Returns
        -------
        jnp.ndarray
            Log-likelihood values with only the Gaussian prior.
        """
        params = self._parametrization.reals1d_to_params(theta_reals1d)
        return self.only_prior(params, **kwargs)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d: jnp.ndarray, **kwargs):
        """Compute likelihood array with individual components.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Parameters used to the log-likelihood computation.
        **kwargs : dict
            a dict where all additional log_likelihood arguments can be found.

        Returns
        -------
        jnp.ndarray
            Array of log-likelihood components for each individual.
        """
        return self.log_likelihood_without_prior(
            theta_reals1d, **kwargs
        ) + self.log_likelihood_only_prior(theta_reals1d, **kwargs)

    # ============================================================== #
    @abstractmethod
    def sample(self, params_star, prngkey, **kwargs) -> tuple[dict, dict]:
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
        obs, sim = AbstractLatentVariablesModel.sample(self, params_star, prngkey)

        y_without_noise = self.mixed_effect_function(
            params_star, times=kwargs["mem_obs_time"], **sim, **kwargs, **self._cst
        )

        key, prngkey = jrd.split(prngkey, num=2)
        sim["eps"] = jnp.sqrt(params_star.var_residual) * jrd.normal(
            key, shape=y_without_noise.shape
        )

        Y = y_without_noise + sim["eps"]

        return (obs | {"Y": Y}, sim)
