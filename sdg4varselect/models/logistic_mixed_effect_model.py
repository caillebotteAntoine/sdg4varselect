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

from sdg4varselect.models import AbstractMixedEffectsModel


class LogisticMixedEffectsModel(AbstractMixedEffectsModel):
    """define a logistic mixed effects model"""

    def __init__(self, N=1, J=1, **kwargs):
        super().__init__(
            N=N,
            J=J,
            me_name=["phi1", "phi2"],
            **kwargs,
        )

        # self._parametrization = pc.NamedTuple(
        #     mean_latent=pc.NamedTuple(
        #         mu1=pc.RealPositive(scale=0.5),
        #         mu2=pc.Real(loc=50, scale=100),
        #     ),
        #     mu3=pc.RealPositive(scale=5),
        #     cov_latent=pc.MatrixSymPosDef(dim=2, scale=(0.001, 10)),
        #     var_residual=pc.RealPositive(scale=0.001),
        # )
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                mu1=pc.RealPositive(scale=100),
                mu2=pc.Real(loc=100, scale=100),
            ),
            mu3=pc.RealPositive(scale=100),
            cov_latent=pc.MatrixSymPosDef(dim=2, scale=(200, 200)),
            var_residual=pc.RealPositive(scale=100),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi1: jnp.ndarray,  # shape = (N,) [:,None]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        """

        out = phi1[:, None] / (1 + jnp.exp(-(times - phi2[:, None]) / params.mu3))
        assert out.shape == times.shape
        return out

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
        time = jnp.linspace(100, 1500, self.J)
        time = jnp.tile(time, (self.N, 1))
        time += 10 * jrd.uniform(prngkey_time, minval=-2, maxval=2, shape=time.shape)

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time
        )

        return {"mem_obs_time": time} | obs, sim


# ============================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myModel = LogisticMixedEffectsModel(N=1000, J=5)

    p_star = myModel.new_params(
        mean_latent={"mu1": 0.3, "mu2": 90.0},
        mu3=7.5,
        cov_latent=jnp.diag(jnp.array([0.0025, 20])),
        var_residual=0.001,
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.figure()
    plt.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")

    jrd.normal(jrd.PRNGKey(0), shape=(100, 2))

    # myModel = LogisticMixedEffectsModel(N=20, J=8)

    # my_params_star = myModel.new_params(
    #     mu1=20,
    #     mu2=90.0,
    #     mu3=7.5,
    #     gamma2_1=0.25,
    #     gamma2_2=20,
    #     sigma2=0.001,
    # )

    # myobs, mysim = myModel.sample(my_params_star, jrd.PRNGKey(0))

    # my_params_star = myModel.new_params(
    #     mu1=20,
    #     mu2=90.0,
    #     mu3=7.5,
    #     gamma2_1=0.25,
    #     gamma2_2=20,
    #     sigma2=0.001,
    # )
    # myobs, mysim = myModel.sample(my_params_star, jrd.PRNGKey(0))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")

    # my_params_star = myModel.new_params(
    #     mu1=15,
    #     mu2=90.0,
    #     mu3=7.5,
    #     gamma2_1=0.25,
    #     gamma2_2=20,
    #     sigma2=0.001,
    # )
    # myobs, mysim = myModel.sample(my_params_star, jrd.PRNGKey(0))
    # ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")

    # ax.set_ylabel("Time from flowering")
    # ax.set_xlabel("Growing degree day")
    # # ax.set_xticks([70, 100, 130], ["juil.", "août", "sept."])
    # ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # ax.yaxis.get_label().set_fontsize(20)
    # ax.xaxis.get_label().set_fontsize(20)
