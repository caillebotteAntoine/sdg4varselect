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


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


class AbstractMixedEffectsModel(AbstractModel):
    def __init__(self, N, J, **kwargs):
        AbstractModel.__init__(self, N, **kwargs)

        self._j = J

    @property
    def J(self):
        return self._j

    # ============================================================== #
    @abstractmethod
    def mixed_effect_function(self, params, *args, **kwargs):
        """Function that return an non linear fct that define the mixed effect models"""

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""

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

        likelihood_mem = -J / 2 * jnp.log(2 * jnp.pi * params.sigma2) - jnp.nansum(
            (Y - pred) ** 2, axis=1
        ) / (2 * params.sigma2)

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
