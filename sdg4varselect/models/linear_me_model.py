"""
Module for abstract class AbstractJointModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221

import functools


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
import parametrization_cookbook.jax as pc

from sdg4varselect.models.abstract.abstract_mixed_effects_model import (
    AbstractMixedEffectsModel,
    mem_simulation,
)


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


class LinearLatentModelH1(AbstractMixedEffectsModel):
    """
    Y = intercept + slope*t

    Y-Yobs
    """

    def __init__(self, N, J, **kwargs):
        AbstractMixedEffectsModel.__init__(self, N, J, **kwargs)

        self._parametrization = pc.NamedTuple(
            mu2=pc.RealPositive(scale=1),
            gamma2_2=pc.RealPositive(scale=1),
            sigma2=pc.RealPositive(scale=1),
        )

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"LEM_N{self.N}"

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        """

        return phi2[:, None] * times

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""
        latent_prior = gaussian_prior(
            kwargs["phi2"],
            params.mu2,
            params.gamma2_2,
        )
        return latent_prior

    # ================== DATA GENERATION ================== #
    def sample(
        self,
        params_star,
        prngkey,
    ):
        (
            prngkey_time,
            prngkey_mem,
        ) = jrd.split(prngkey, num=2)

        obs, sim = {}, {}

        time = jnp.repeat(jnp.linspace(0, 2, num=self.J)[None, :], self.N, axis=0)

        obs = {"mem_obs_time": time}

        obs_mem, sim = mem_simulation(
            params_star,
            prngkey_mem,
            n_ind=self.N,
            noise_variance="sigma2",
            fct=self.mixed_effect_function,
            random_effects={
                "phi2": ("mu2", "gamma2_2"),
            },
            fct_kwargs={"times": obs["mem_obs_time"]},
        )
        obs.update(obs_mem)
        return obs, sim


class LinearLatentModelH0(AbstractMixedEffectsModel):
    """
    Y = intercept + slope*t

    Y-Yobs
    """

    def __init__(self, N, J, **kwargs):
        AbstractMixedEffectsModel.__init__(self, N, J, **kwargs)

        self._parametrization = pc.NamedTuple(
            gamma2_2=pc.RealPositive(scale=1),
            sigma2=pc.RealPositive(scale=1),
        )

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"LEM_N{self.N}"

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        """

        return phi2[:, None] * times

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""
        latent_prior = gaussian_prior(
            kwargs["phi2"],
            0,
            params.gamma2_2,
        )
        return latent_prior

    # ================== DATA GENERATION ================== #
    def sample(
        self,
        params_star,
        prngkey,
    ):
        (
            prngkey_time,
            prngkey_mem,
        ) = jrd.split(prngkey, num=2)

        obs, sim = {}, {}

        time = jnp.repeat(jnp.linspace(0, 2, num=self.J)[None, :], self.N, axis=0)

        obs = {"mem_obs_time": time}

        obs_mem, sim = mem_simulation(
            params_star,
            prngkey_mem,
            n_ind=self.N,
            noise_variance="sigma2",
            fct=self.mixed_effect_function,
            random_effects={
                "phi2": ("mu2", "gamma2_2"),
            },
            fct_kwargs={"times": obs["mem_obs_time"]},
        )
        obs.update(obs_mem)
        return obs, sim


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myModel = LinearLatentModelH1(N=100, J=5)

    p_star = myModel.new_params(
        mu2=2,
        gamma2_2=0.1,
        sigma2=0.1,
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.figure()
    plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")

    myModel = LinearLatentModelH0(N=100, J=5)

    p_star = myModel.new_params(
        gamma2_2=0.1,
        sigma2=0.1,
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.figure()
    plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
