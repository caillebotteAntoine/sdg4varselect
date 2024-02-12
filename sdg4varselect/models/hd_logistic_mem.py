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

from sdg4varselect.models.abstract_mixed_effect_model import (
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
        super().__init__(**kwargs)
        self._N = N
        self._J = J
        self._P = P

        self._parametrization = pc.NamedTuple(
            eta1=pc.RealPositive(scale=300),
            eta2=pc.RealPositive(scale=300),
            gamma2_1=pc.RealPositive(scale=0.1),
            gamma2_2=pc.RealPositive(scale=0.1),
            mu=pc.RealPositive(scale=1500),
            sigma2=pc.RealPositive(scale=50),
            Gamma2=pc.RealPositive(scale=300),
            beta=pc.RealPositive(scale=100, shape=(P,)),
        )

    @property
    def N(self):
        return self._N

    @property
    def J(self):
        return self._J

    @property
    def P(self):
        return self._P

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        psi1: jnp.ndarray,  # shape = (N,) [:,None]
        psi2: jnp.ndarray,  # shape = (N,) [:,None]
        ksi: jnp.ndarray,  # shape = (N,) [:,None]
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        psi1 = supremum
        psi2 = midpoint
        """

        phi = params.mu + cov @ params.beta + ksi
        out = psi1[:, None] / (1 + jnp.exp(-(times - phi[:, None]) / psi2[:, None]))
        assert out.shape == times.shape
        return out

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_mem_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""
        latent_prior = gaussian_prior(
            kwargs["psi1"],
            params.eta1,
            params.gamma2_1,
        ) + gaussian_prior(
            kwargs["psi2"],
            params.eta2,
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
            prngkey_cov,
        ) = jrd.split(prngkey, num=3)

        # === nlmem_simulation() === #
        time = 150 + jnp.arange(0, self.J - 1) * (3000 - 150) / (self.J - 1)
        time = jnp.repeat(time[None, :], self.N, axis=0)

        obs = {"mem_obs_time": time}
        cov = cov_simulation(prngkey_cov, min=-1, max=1, shape=(self.N, self.P))

        obs_mem, sim = mem_simulation(
            params_star,
            prngkey_mem,
            n_ind=self.N,
            noise_variance="sigma2",
            fct=self.mixed_effect_function,
            random_effects={
                "psi1": ("eta1", "gamma2_1"),
                "psi2": ("eta2", "gamma2_2"),
                "ksi": (0, "gamma2_2"),
            },
            fct_kwargs={"times": obs["mem_obs_time"], "cov": cov},
        )

        obs.update(obs_mem)
        obs.update({"cov": cov})

        return obs, sim


# ============================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myModel = HDLogisticMixedEffectsModel(N=100, J=10, P=5)

    my_params_star = myModel.new_params(
        eta1=200,
        eta2=300,
        gamma2_1=0.1,
        gamma2_2=0.1,
        mu=1200,
        sigma2=30,
        Gamma2=200,
        beta=jnp.concatenate(
            [jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]
        ),
    )

    myobs, mysim = myModel.sample(my_params_star, jrd.PRNGKey(0))

    plt.figure()
    plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
