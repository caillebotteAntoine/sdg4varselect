"""
Module for abstract class AbstractMixedEffectsModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

from abc import abstractmethod
import functools


import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    AbstractLatentVariablesModel,
)


class AbstractMixedEffectsModel(AbstractModel, AbstractLatentVariablesModel):
    """the most abstact model with mixed effects model that can be defined"""

    def __init__(
        self,
        N: int,
        J: int,
        me_name: list[str],
        **kwargs,
    ):
        AbstractModel.__init__(self, N, **kwargs)
        AbstractLatentVariablesModel.__init__(
            self, me_name, me_size=[self.N for _ in me_name]
        )

        self._j = J

    @property
    def J(self):
        return self._j

    # ============================================================== #
    @abstractmethod
    def mixed_effect_function(self, params, *args, **kwargs):
        """Function that return an non linear fct that define the mixed effect models"""

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_without_prior(
        self, params, Y, mem_obs_time, **kwargs
    ) -> jnp.ndarray:
        """return likelihood without the gaussian prior"""
        N, J = Y.shape
        assert mem_obs_time.shape == (N, J)
        # assert phi1.shape == (N,)
        # assert phi2.shape == (N,)

        pred = self.mixed_effect_function(
            params, mem_obs_time, **self._cst, **kwargs
        )  # shape = (N,J)

        # mise à jours de J en fct des nan ?!
        likelihood_mem = -J / 2 * jnp.log(
            2 * jnp.pi * params.var_residual
        ) - jnp.nansum((Y - pred) ** 2, axis=1) / (2 * params.var_residual)

        assert likelihood_mem.shape == (N,)
        return likelihood_mem

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_array(self, theta_reals1d, **kwargs):
        """return likelihood as array each component for each individuals"""
        params = self._parametrization.reals1d_to_params(theta_reals1d)

        return self.likelihood_without_prior(
            params, **kwargs
        ) + self.likelihood_only_prior(params, **kwargs)

    # ============================================================== #
    @abstractmethod
    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ):
        """Sample one data set for the model"""

        key, prngkey = jrd.split(prngkey, num=2)

        D = len(self.latent_variables_name)
        assert len(params_star.mean_latent) == D

        sim_latent = self.sample_normal(key, params_star, shape=(self.N, D))

        sim = dict(
            zip(
                self.latent_variables_name,
                [sim_latent[:, i] for i in range(D)],
            )
        )

        y_without_noise = self.mixed_effect_function(
            params_star, times=kwargs["mem_obs_time"], **sim
        )

        key, prngkey = jrd.split(prngkey, num=2)
        sim["eps"] = jnp.sqrt(params_star.var_residual) * jrd.normal(
            key, shape=y_without_noise.shape
        )

        Y = y_without_noise + sim["eps"]

        return (
            {"Y": Y},  # obs
            sim,
        )


# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #


def mem_simulation(
    params,
    prngkey,
    n_ind,
    noise_variance,  # = sigma2
    fct,  # = logistic_curve
    random_effects,  # = { "phi1":("mu1", "gamma2_1"), "phi2" }
    fct_kwargs,  # other parameters, example = [time]
):
    """return simulation following mixed effect model
    Y = fct(random_effects, fixed_effects,kwargs) + N(0, noise_variance^2)
    """

    sim = {}
    for name, value in random_effects.items():
        key, prngkey = jrd.split(prngkey, num=2)

        if value[0] == 0:
            mean = 0
        else:
            mean = getattr(params, value[0])
        var = getattr(params, value[1])
        # N(mean, var^2)
        sim[name] = mean + jnp.sqrt(var) * jrd.normal(key, shape=(n_ind,))
        fct_kwargs[name] = sim[name]

    y_without_noise = fct(params, **fct_kwargs)

    key, prngkey = jrd.split(prngkey, num=2)
    sim["eps"] = jnp.sqrt(getattr(params, noise_variance)) * jrd.normal(
        key, shape=y_without_noise.shape
    )

    Y = y_without_noise + sim["eps"]

    return (
        {"Y": Y},  # obs
        sim,
    )
