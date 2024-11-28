"""
Module for the abstract class `AbstractCoxMemJointModel`.

This module defines an abstract class `AbstractCoxMemJointModel`, which represents
a joint model combining a Cox model and a mixed effects model.

Created by antoine.caillebotte@inrae.fr
"""

import functools
import parametrization_cookbook.jax as pc

import jax.numpy as jnp
import jax.random as jrd
from jax import jit


from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    AbstractLatentVariablesModel,
)
from sdg4varselect.models.abstract.abstract_cox_model import AbstractCoxModel
from sdg4varselect.models.abstract.abstract_mixed_effects_model import (
    AbstractMixedEffectsModel,
)
from copy import deepcopy

# pylint: disable = all


class AbstractCoxMemJointModel(AbstractCoxModel, AbstractLatentVariablesModel):
    """Abstract class defining a joint model of a mixed effects model and a Cox model.

    This class combines a Cox proportional hazards model with a mixed effects model
    for survival analysis with latent variables. It integrates both models into a
    single joint model framework.

    Parameters
    ----------
    mem : type[AbstractMixedEffectsModel]
        Class for the mixed effects model.
    cox : type[AbstractCoxModel]
        Class for the Cox model.
    alpha_scale : float
        Scaling factor for the alpha parameter.
    **kwargs : dict
        Additional keyword arguments for model initialization.
    """

    def __init__(
        self,
        mem: type[AbstractMixedEffectsModel],
        cox: type[AbstractCoxModel],
        alpha_scale,
        **kwargs,
    ):
        # assert mem.N == cox.N
        self._alpha_scale = alpha_scale
        self._mem = deepcopy(mem)
        self._cox = deepcopy(cox)
        AbstractCoxModel.__init__(self, N=cox.N, P=cox.P, **kwargs)
        AbstractLatentVariablesModel.__init__(
            self, self._mem.latent_variables_name, self._mem.latent_variables_size
        )

    def init_parametrization(self):
        params = self._mem.parametrization._params | self._cox._parametrization._params
        self._parametrization = pc.NamedTuple(
            alpha=pc.Real(scale=self._alpha_scale), **params
        )
        self._mem._parametrization = self._parametrization
        self._cox._parametrization = self._parametrization
        self._cst |= self._mem._cst | self._cox._cst
        self._mem._cst = self._cst
        self._cox._cst = self._cst
        print("parametrization have been merged !")

    @property
    def J(self):
        """int: Number of observation times."""
        return self._mem.J

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, times, **kwargs):
        """Define the log of the baseline hazard. Must be implemented by subclasses."""
        return self._cox.log_baseline_hazard(params, times, **kwargs)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def link_function(
        self, alpha, params, times: jnp.ndarray, **kwargs  # shape = (N,num)
    ):
        """Define the link function for the model. Must be implemented by subclasses.

        Parameters
        ----------
        alpha : float
            Alpha parameter for the link function.
        params : dict
            Model parameters.
        times : jnp.ndarray
            Array of surival observations, shape (N, num).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Values of the link function.
        """
        raise NotImplementedError

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_hazard(
        self,
        params,
        times: jnp.ndarray,  # shape = (N,num)
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N, num)
        """Compute the log of the hazard function.

        Parameters
        ----------
        params : dict
            Model parameters.
        times : jnp.ndarray
            Array of time points, shape (N, num).
        cov : jnp.ndarray
            Covariates matrix, shape (N, p).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Log hazard values, shape (N, num).

        Notes
        -----
        hazard(t) = h0(t) * exp(\beta^T U +f(\alpha,params, t)))

        log(h(t)) = log(h0(t))+\beta^T U +f(\alpha,params, t))
        """
        link_values = self.link_function(params.alpha, params, times, **kwargs)
        log_h = self._cox.log_hazard(params, times, cov, **kwargs)
        # print(params.alpha)
        return log_h + link_values

    # ============================================================== #
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
        return self._mem.log_likelihood_only_prior(theta_reals1d, **kwargs)

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
        mem_log_likelihood_array = self._mem.log_likelihood_without_prior(
            theta_reals1d, **kwargs
        )
        cox_log_likelihood_array = AbstractCoxModel.log_likelihood_array(
            self, theta_reals1d, **kwargs
        )
        return cox_log_likelihood_array
        return mem_log_likelihood_array + cox_log_likelihood_array

    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d, **kwargs):
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
    def sample(
        self, params_star, prngkey, simulation_intervalle, **kwargs
    ) -> tuple[dict, dict]:
        """
        Sample a dataset from the joint model.

        Parameters
        ----------
        params_star : dict
            Model parameters for sampling.
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        simulation_intervalle : tuple
            Interval within which to simulate times.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
                - dict: Generated observations.
                - dict: Simulated variables.
        """

        prngkey_mem, prngkey_cox = jrd.split(prngkey)
        obs, sim = self._mem.sample(params_star, prngkey_mem)

        coxobs, coxsim = AbstractCoxModel.sample(
            self,
            params_star,
            prngkey_cox,
            simulation_intervalle,
            **kwargs,
            **sim,
        )

        return obs | coxobs, sim | coxsim

    # ============================================================== #
    def censoring_simulation(self, prngkey, T, params_star, **kwargs):
        return self._cox.censoring_simulation(prngkey, T, params_star, **kwargs)

    def covariates_simulation(self, prngkey, **kwargs) -> jnp.ndarray:
        return self._cox.covariates_simulation(prngkey, **kwargs)
