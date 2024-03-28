"""
Module for abstract class AbstractLatentVariablesModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

import functools
from jax import jit
import jax.numpy as jnp
import jax.random as jrd


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


# @jit
def log_gaussian_prior_cov(
    data: jnp.ndarray,  # shape = (N,D)
    mean: jnp.ndarray,  # shape = (D,)
    cov: jnp.ndarray,  # shape = (D,D)
) -> jnp.ndarray:
    """Computation of the current target gaussian prior with covariance matrix

    sqrt((2pi)^D det(cov)) * exp[-1/2(x-m)^T cov^-1 (x-m)]

    Parameters:
    ----------
        data (jnp.ndarray): observed data of shape (N,J),
        mean (jnp.ndarray): mean of the latente variables of shape (D,),
        cov (jnp.ndarray): covariance matrix of the latente variables of shape (D,D).
    """

    data_sub_mean = data - mean
    D = mean.shape[0]

    out = (
        jnp.linalg.slogdet(cov)[1]  # log du det
        + D * jnp.log(2 * jnp.pi)
        + ((data_sub_mean @ jnp.linalg.inv(cov)) * data_sub_mean).sum(axis=1)
    )

    return -out / 2


class AbstractLatentVariablesModel:
    """the most abstact model with latent variables that can be defined

    latent variables parameter must be named mean_latent and cov_latent

    mean_latent can be Scalar parametrization of Shape (1,), Tuple or Namedtuple
    of Scalar parametrization
    """

    def __init__(
        self,
        me_name: list[str],
        me_size=list[int],
        **kwargs,
    ):
        self._latent_variables_name = me_name

        self._latent_variables_size = me_size

    @property
    def latent_variables_name(self):
        return self._latent_variables_name

    def latent_variables_data(self, params, name):
        i = self._latent_variables_name.index(name)

        return {
            "size": self._latent_variables_size[i],
            "mean": float(params.mean_latent[i]),
        }

    # @functools.partial(jit, static_argnums=(0, 1))
    # def get_mean_var(self, name, params):
    #     mean = self._mixed_effect[name]["mean"]
    #     variance = self._mixed_effect[name]["variance"]

    #     return (
    #         0 if mean is None else params.__getattribute__(mean),
    #         params.__getattribute__(variance),
    #     )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""

        data = [kwargs[name] for name in self._latent_variables_name]

        return log_gaussian_prior_cov(
            data=jnp.array(data).T,
            mean=jnp.array(params.mean_latent),
            cov=params.cov_latent,
        )

        # latent_prior = [
        #     gaussian_prior(kwargs[name], *self.get_mean_var(name, params))
        #     for name in self._latent_variables_name
        # ]
        # return jnp.array(latent_prior).sum(axis=0)

    def sample_normal(self, prngkey, params, N):
        # mean, var = self.get_mean_var(name, params)
        # return mean + jnp.sqrt(var) * jrd.normal(prngkey, shape=shape)

        mean = jnp.array(params.mean_latent)
        shape = (N, mean.shape[0])
        return (
            mean
            + jrd.normal(prngkey, shape=shape)
            @ jnp.linalg.cholesky(params.cov_latent).T
        )
