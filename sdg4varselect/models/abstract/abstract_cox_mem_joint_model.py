"""
Module for the abstract class `AbstractCoxMemJointModel`.

This module defines an abstract class `AbstractCoxMemJointModel`, which represents
a joint model combining a Cox model and a mixed effects model.

Created by antoine.caillebotte@inrae.fr
"""

import functools
from copy import deepcopy

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


# pylint: disable = protected-access


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
        AbstractCoxModel.__init__(self, N=mem.N, P=cox.P, **kwargs)
        AbstractLatentVariablesModel.__init__(self, self._mem.latent_variables_size)
        for name in self._mem.latent_variables_name:
            self.add_latent_variables(name)

    def init_parametrization(self):
        self._mem.init_parametrization()
        self._cox.init_parametrization()

        params = (
            self._mem.parametrization._params
            | {"alpha": pc.Real(scale=self._alpha_scale)}
            | self._cox._parametrization._params
        )
        self._parametrization = pc.NamedTuple(**params)
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

    @property
    def P(self) -> int:
        """Get the number of high-dimensional parameters.

        Returns
        -------
        int
            Number of high-dimensional parameters in the model.
        """
        return self._p

    @P.setter
    def P(self, P):
        self._p = int(P)
        self._cox.P = self._p

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def proportional_hazards_component(self, params, **kwargs):
        """Compute the proportional hazards component.

        Parameters
        ----------
        params : dict
            Model parameters.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Proportional hazards component.
        """
        phc = self._cox.proportional_hazards_component(params, **kwargs)
        link_values = self.link_function(params.alpha, params, **kwargs)
        assert phc.shape[0] == link_values.shape[0]
        assert phc.shape[1] == link_values.shape[1] or phc.shape[1] == 1

        return phc + link_values

    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, **kwargs):
        """Compute the log baseline hazard.

        Parameters
        ----------
        params : dict
            Model parameters.
        **kwargs : dict
            Additional parameters.
        Returns
        -------
        jnp.ndarray
            Log baseline hazard.
        """
        return self._cox.log_baseline_hazard(params, **kwargs)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def link_function(
        self,
        alpha,
        params,
        survival_int_range: jnp.ndarray,
        **kwargs,  # shape = (N,num)
    ):
        """Define the link function for the model. Must be implemented by subclasses.

        Parameters
        ----------
        alpha : float
            Alpha parameter for the link function.
        params : dict
            Model parameters.
        survival_int_range : jnp.ndarray
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
        self, params_star, prngkey, linspace_num=100000, **kwargs
    ) -> tuple[dict, dict]:
        """
        Sample a dataset from the joint model.

        Parameters
        ----------
        params_star : dict
            Model parameters for sampling.
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        linspace_num : int, optional
            Number of points in the linspace for survival interval range, by default 100000.
        **kwargs : dict
            Additional parameters.
            containing simulation_intervalle : tuple, Interval within which to simulate times.

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
            linspace_num=linspace_num,
            **kwargs,
            **sim,
        )

        ii_notobserved = obs["mem_obs_time"] <= coxobs["T"][:, None]
        obs["Y"] = jnp.where(ii_notobserved, obs["Y"], jnp.nan)
        obs["mem_obs_time"] = jnp.where(ii_notobserved, obs["mem_obs_time"], jnp.nan)

        return obs | coxobs, sim | coxsim

    # ============================================================== #
    def censoring_simulation(self, prngkey, T, params_star, **kwargs):
        """Simulate censoring times for the Cox model.
        Parameters
        ----------
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        T : jnp.ndarray
            Event times.
        params_star : dict
            Model parameters for simulation.
        **kwargs : dict
            Additional parameters for censoring simulation.
        Returns
        -------
        jnp.ndarray
            Simulated censoring times.
        """
        return self._cox.censoring_simulation(prngkey, T, params_star, **kwargs)

    def covariates_simulation(self, prngkey, **kwargs) -> jnp.ndarray:
        """Simulate covariates for the Cox model.
        Parameters
        ----------
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        **kwargs : dict
            Additional parameters for covariate simulation.
        Returns
        -------
        jnp.ndarray
            Simulated covariates.
        """
        return self._cox.covariates_simulation(prngkey, **kwargs)
