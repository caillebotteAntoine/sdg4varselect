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

from sdg4varselect.models.abstract.abstract_mixed_effect_model import (
    AbstractMixedEffectsModel,
    mem_simulation,
)


def cov_simulation(prngkey, min, max, shape):
    cov = jrd.uniform(prngkey, minval=min, maxval=max, shape=shape)
    cov = cov - cov.mean(axis=0)[None, :]
    cov = jnp.array(cov, dtype=jnp.float32)

    return cov


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


class HDLogisticMixedEffectsModel(AbstractMixedEffectsModel):
    def __init__(self, N=1, J=1, P=1, **kwargs):
        super().__init__(N=N, J=J, **kwargs)
        self._P = P

        self._parametrization = pc.NamedTuple(
            mu1=pc.RealPositive(scale=0.5),
            mu2=pc.RealPositive(scale=100),
            mu3=pc.RealPositive(scale=5),
            gamma2_1=pc.RealPositive(scale=0.001),
            gamma2_2=pc.RealPositive(scale=10),
            sigma2=pc.RealPositive(scale=0.001),
            beta=pc.Real(scale=10, shape=(P,)),
        )

    @property
    def P(self):
        return self._P

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi1: jnp.ndarray,  # shape = (N,) [:,None]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        """

        ksi = cov @ params.beta + phi2
        out = phi1[:, None] / (1 + jnp.exp(-(times - ksi[:, None]) / params.mu3))
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

        (prngkey_time, prngkey_mem, prngkey_cov) = jrd.split(prngkey, num=3)

        # === nlmem_simulation() === #
        time = jnp.repeat(jnp.linspace(60, 135, num=self.J)[None, :], self.N, axis=0)
        time += jrd.uniform(prngkey_time, minval=-2, maxval=2, shape=time.shape)

        obs = {"mem_obs_time": time}
        cov = cov_simulation(prngkey_cov, min=-1, max=1, shape=(self.N, self.P))

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
            fct_kwargs={"times": obs["mem_obs_time"], "cov": cov},
        )
        obs.update(obs_mem)
        obs.update({"cov": cov})

        return obs, sim


# ============================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myModel = HDLogisticMixedEffectsModel(N=100, J=5, P=10)

    p_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        beta=jnp.concatenate(
            [jnp.array([-3, -2, 2, 3]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.figure()
    plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
