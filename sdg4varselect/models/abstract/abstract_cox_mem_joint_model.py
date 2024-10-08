"""
Module for abstract class AbstractCoxModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0212

import functools
import parametrization_cookbook.jax as pc

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models.abstract.abstract_cox_model import (
    AbstractCoxModel,
)
from sdg4varselect.models.abstract.abstract_mixed_effects_model import (
    AbstractMixedEffectsModel,
)


class AbstractCoxMemJointModel(AbstractCoxModel):
    """define a joint model of an mixed effects model and a cox model"""

    def __init__(self, mem: type[AbstractMixedEffectsModel], P, **kwargs):
        AbstractCoxModel.__init__(self, N=mem.N, P=P, **kwargs)

        self._mem = mem

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""
        params = self._mem.parametrization._params | self.parametrization._params
        self._parametrization = pc.NamedTuple(**params)
        self._mem._parametrization = pc.NamedTuple(**params)
        self._cst |= self._mem._cst

        AbstractCoxModel.init(self)

    @property
    def J(self):
        return self._mem.J

    @property
    def latent_variables_name(self):
        return self._mem._latent_variables_name

    def latent_variables_data(self, params, name):
        return self._mem.latent_variables_data(params, name)

    # @property
    # def name(self):
    #     """return a str called name, based on the parameter of the model"""
    #     return f"WCoxJM_N{self.N}_J{self.J}_P{self.DIMCovCox}"

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def link_function(
        self,
        alpha: jnp.ndarray,  # shape = (1,),
        params,
        times: jnp.ndarray,  # shape = (N,num)
        **kwargs,
    ):
        mem_value = self._mem.mixed_effect_function(params, times, **kwargs)
        assert mem_value.shape == times.shape
        return alpha * mem_value

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_hazard(
        self,
        params,
        times: jnp.ndarray,  # shape = (N,num)
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N, num)
        """hazard(t) = h0(t) * exp(beta^T U )
        with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}tu com

        return : log(b/a) + (b-1)*log(t/a) + beta^T U
        """
        link_values = self.link_function(params.alpha, params, times, **kwargs)
        log_h = AbstractCoxModel.log_hazard(self, params, times, cov, **kwargs)
        return log_h + link_values

    @functools.partial(jit, static_argnums=0)
    def log_likelihood_without_prior(self, params, **kwargs):

        mem_log_likelihood = self._mem.log_likelihood_without_prior(params, **kwargs)
        cox_log_likelihood = AbstractCoxModel.log_likelihood_without_prior(
            self, params, **kwargs
        )

        return mem_log_likelihood + cox_log_likelihood

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return log likelihood with only the gaussian prior"""
        return self._mem.log_likelihood_only_prior(params, **kwargs)

    # ============================================================== #
    def sample(
        self,
        params_star,
        prngkey,
        weibull_censoring_loc,
        **kwargs,
    ):

        prngkey_mem, prngkey_cox = jrd.split(prngkey)
        obs, sim = self._mem.sample(params_star, prngkey_mem)

        coxobs, coxsim = AbstractCoxModel.sample(
            self, params_star, prngkey_cox, weibull_censoring_loc, **sim
        )

        return obs | coxobs, sim | coxsim

    def sample_normal(self, *args, **kwargs):
        return self._mem.sample_normal(*args, **kwargs)
