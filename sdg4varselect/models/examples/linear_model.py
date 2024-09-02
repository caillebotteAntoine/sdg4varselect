# pylint: disable=W0221
import functools
import parametrization_cookbook.jax as pc

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models.abstract.abstract_model import AbstractModel


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


class LinearModel(AbstractModel):
    """
    Y = intercept + slope*t

    Y-Yobs
    """

    def __init__(self, N, **kwargs):
        AbstractModel.__init__(self, N, **kwargs)

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            intercept=pc.Real(scale=1),
            slope=pc.Real(scale=1),
            sigma2=pc.RealPositive(scale=0.1),
        )

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"LM_N{self.N}"

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d, Y, time, **kwargs):
        """return likelihood"""
        params = self._parametrization.reals1d_to_params(theta_reals1d)

        return gaussian_prior(
            data=Y,
            mean=params.intercept + params.slope * time,
            variance=params.sigma2,
        )

    # ================== DATA GENERATION ================== #
    def sample(
        self,
        params_star,
        prngkey,
    ):
        obs, sim = {}, {}

        obs["time"] = jnp.linspace(0, 2, num=self.N)
        Y_without_noise = params_star.intercept + params_star.slope * obs["time"]

        key, prngkey = jrd.split(prngkey, num=2)
        sim["eps"] = jnp.sqrt(params_star.sigma2) * jrd.normal(
            key, shape=Y_without_noise.shape
        )

        obs["Y"] = Y_without_noise + sim["eps"]

        return obs, sim


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myModel = LinearModel(150)
    my_params_star = myModel.new_params(intercept=1.5, slope=0.5, sigma2=0.1)

    myobs, _ = myModel.sample(my_params_star, jrd.PRNGKey(0))

    plt.plot(myobs["time"], myobs["Y"], ".")

    theta0 = jrd.normal(jrd.PRNGKey(0), shape=(myModel.parametrization.size,))

    myModel.likelihood_array(theta0, **myobs)
