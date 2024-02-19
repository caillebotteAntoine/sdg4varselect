"""
Module for abstract class AbstractJointModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221

from abc import abstractmethod
import functools

import jax.numpy as jnp
import jax.random as jrd
from jax import jit
import parametrization_cookbook.jax as pc

from sdg4varselect.models import (
    AbstractMixedEffectsModel,
    mem_simulation,
    gaussian_prior,
)


class PharmacoKineticMixedEffectsModel(AbstractMixedEffectsModel):
    """define a Pharmaco Kinetic mixed effects model"""

    def __init__(self, N=1, J=1, **kwargs):
        super().__init__(N=N, J=J, D=-100, **kwargs)

        self._parametrization = pc.NamedTuple(
            mu1=pc.RealPositive(scale=10),
            mu2=pc.RealPositive(scale=5),
            mu3=pc.RealPositive(scale=50),
            gamma2_1=pc.RealPositive(scale=1),
            gamma2_2=pc.RealPositive(scale=1),
            sigma2=pc.RealPositive(scale=0.001),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi1: jnp.ndarray,  # shape = (N,) [:,None]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        D,
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        """
        out = (
            D
            * phi1[:, None]
            / (params.mu3 * (phi1[:, None] - phi2[:, None] / params.mu3))
            * (
                jnp.exp(-phi1[:, None] * times)
                - jnp.exp(-phi2[:, None] / params.mu3 * times)
            )
        )
        assert out.shape == times.shape
        return out

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""
        latent_prior = gaussian_prior(
            kwargs["phi1"],
            params.mu1,
            params.gamma2_1,
        ) + gaussian_prior(
            kwargs["phi2"],
            params.mu2,
            params.gamma2_2,
        )
        return latent_prior

    # ============================================================== #

    @abstractmethod
    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ):
        """Sample one data set for the model"""

        (
            prngkey_time,
            prngkey_mem,
        ) = jrd.split(prngkey, num=2)

        # === nlmem_simulation() === #
        time = jnp.repeat(
            jnp.exp(jnp.linspace(-3, 4, num=self.J))[jnp.newaxis, :],
            self.N,
            axis=0,
        )
        # time += jrd.uniform(prngkey_time, minval=-2, maxval=2, shape=time.shape)

        obs = {"mem_obs_time": time}

        obs_mem, sim = mem_simulation(
            params_star,
            prngkey_mem,
            n_ind=self.N,
            noise_variance="sigma2",
            fct=self.mixed_effect_function,
            random_effects={
                "phi1": ("mu1", "gamma2_1"),
                "phi2": ("mu2", "gamma2_2"),
            },
            fct_kwargs={"times": obs["mem_obs_time"]} | self._cst,
        )
        obs.update(obs_mem)

        return obs, sim


# ============================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myModel = PharmacoKineticMixedEffectsModel(N=100, J=15)

    p_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.figure()
    _ = plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
