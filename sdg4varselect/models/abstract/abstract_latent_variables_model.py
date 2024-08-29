"""
Module for abstract class AbstractLatentVariablesModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

import functools
from typing import Union, Type

from jax import jit
import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect.models.abstract.abstract_model import AbstractModel


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


def sample_normal(prngkey, params, N):
    mean = jnp.array(params.mean_latent)
    shape = (N, mean.shape[0])

    return (
        mean
        + jrd.normal(prngkey, shape=shape) @ jnp.linalg.cholesky(params.cov_latent).T
    )


@jit
def log_gaussian_prior_cov(
    x: jnp.ndarray,  # shape = (N,D)
    mean: jnp.ndarray,  # shape = (D,)
    cov: jnp.ndarray,  # shape = (D,D)
) -> jnp.ndarray:
    """Computation of the current target gaussian prior with covariance matrix

    sqrt((2pi)^D det(cov)) * exp[-1/2(x-m)^T cov^-1 (x-m)]

    Parameters:
    ----------
        x (jnp.ndarray): observed data of shape (N,D),
        mean (jnp.ndarray): mean of the latente variables of shape (D,),
        cov (jnp.ndarray): covariance matrix of the latente variables of shape (D,D).
    """
    N, D = x.shape
    assert mean.shape == (D,)
    assert cov.shape == (D, D)

    x_sub_mean = x - mean

    out = (
        jnp.linalg.slogdet(cov)[1]  # log du det
        + D * jnp.log(2 * jnp.pi)
        + ((x_sub_mean @ jnp.linalg.inv(cov)) * x_sub_mean).sum(axis=1)
    )

    assert out.shape == (N,)
    return -out / 2


if __name__ == "__main__":

    dt = 12 + 3 * jrd.normal(jrd.PRNGKey(0), shape=(1000, 1))

    def f(data, cov, mean):
        return log_gaussian_prior_cov(
            data,
            jnp.array([mean]),
            jnp.array([[cov]]),
        ).sum()

    import matplotlib.pyplot as plt

    xx = jnp.linspace(1, 100, num=1000)

    y = jnp.array([f(dt, k, 12) for k in xx])
    plt.plot(xx, y)

    def g(cov):
        res = 0
        for _ in range(100):
            data = 12 + 3 * jrd.normal(jrd.PRNGKey(0), shape=(1000, 1))
            res += f(data, cov, 12)

        return res

    plt.figure()

    y = jnp.array([g(k) for k in xx])
    plt.plot(xx, y)


class AbstractLatentVariablesModel:
    """the most abstact model with latent variables that can be defined

    latent variables parameter must be named mean_latent and cov_latent.

    Latent variables must be arranged in a precise order:
        those with a zero mean, then those with a non-zero mean.
    All latent variables are defined with a variance and covariance defined in
    cov_latent as follows:

    -   cov_latent must have a Matrix parametrization type.
        cov_latent will have a shape (D,D).

    -   mean_latent can be Scalar parametrization of Shape (1,),
        Tuple or Namedtuple of Scalar parametrization.

        mean_latent shape can't exceed (D,)


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

    @functools.partial(jit, static_argnums=0)
    def mean_and_cov_latent(self, params):
        mean = jnp.array(params.mean_latent)

        cov = params.cov_latent

        z = jnp.zeros(shape=(cov.shape[0] - mean.shape[0],))
        mean = jnp.concatenate([z, mean])
        return cov, mean

    def latent_variables_data(self, params, name):
        i = self._latent_variables_name.index(name)
        _, mean = self.mean_and_cov_latent(params)

        return {
            "size": self._latent_variables_size[i],
            "mean": float(mean[i]),
        }

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return log likelihood with only the gaussian prior"""

        data = [kwargs[name] for name in self._latent_variables_name]
        cov, mean = self.mean_and_cov_latent(params)

        return log_gaussian_prior_cov(
            x=jnp.array(data).T,
            mean=mean,
            cov=cov,
        )


@functools.partial(jit, static_argnums=0)
def new_likelihood(
    model: Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]],
    sample_key,
    data,
    params,
) -> jnp.ndarray:

    sim_latent = sample_normal(sample_key, params=params, N=model.N)

    var_lat_sample = dict(
        zip(
            model.latent_variables_name,
            [sim_latent[:, i] for i in range(sim_latent.shape[1])],
        )
    )

    return jnp.exp(
        model.log_likelihood_without_prior(
            params, **data, **var_lat_sample
        )  # log(f(Y|phi_sim)) ; shape = (N,)
    )  # f(Y|phi_sim) ; shape = (N,)


def log_likelihood_marginal(
    model: Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]],
    prngkey,
    data,
    theta,
    size=2000,
) -> jnp.ndarray:
    """_summary_

    Args:
        model (Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]]): _description_
        prngkey (_type_): _description_
        data (_type_): _description_
        theta (_type_): _description_
        size (int, optional): _description_. Defaults to 2000.

    Returns:
        log_likelihood_marginal (jnp.ndarray) :

        log[ 1/K\sum_{k=1}^K \prod_{i=1}^n f(Y_i|Z_i^{(k)}) ] =
        \sum_{i=1}^n ( log[\sum_{k=1}^K  f(Y_i|Z_i^{(k)})/K ] )

    """
    params = model.parametrization.reals1d_to_params(theta)

    # data = model.latent_variables_data(params0, new_mcmc_name)
    out = []
    for _ in range(size):
        prngkey, sample_key = jrd.split(prngkey, 2)
        out.append(new_likelihood(model, sample_key, data, params))

    return jnp.log((jnp.array(out) / len(out)).sum(axis=0)).sum()


# def log_likelihood_marginal(
#     model: Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]],
#     prngkey,
#     data,
#     theta,
#     size=2000,
# ) -> jnp.ndarray:
#     params = model.parametrization.reals1d_to_params(theta)

#     # data = model.latent_variables_data(params0, new_mcmc_name)
#     def new_likelihood(sample_key) -> jnp.ndarray:

#         sim_latent = sample_normal(sample_key, params=params, N=model.N)

#         var_lat_sample = dict(
#             zip(
#                 model.latent_variables_name,
#                 [sim_latent[:, i] for i in range(sim_latent.shape[1])],
#             )
#         )

#         return jnp.exp(
#             model.log_likelihood_without_prior(
#                 params, **data, **var_lat_sample
#             )  # log(f(Y|phi_sim)) ; shape = (N,)
#         )

#     out = [new_likelihood(prngkey)]
#     n_simu = 200
#     for _ in range(1, n_simu):
#         prngkey, sample_key = jrd.split(prngkey, 2)
#         out.append(out[-1] + new_likelihood(sample_key))

#     while (
#         n_simu < size and abs(out[-2] / (n_simu - 1) - out[-1] / n_simu).all() >= 1e-3
#     ):
#         prngkey, sample_key = jrd.split(prngkey, 2)
#         out.append(out[-1] + new_likelihood(sample_key))
#         n_simu += 1

#     # print(n_simu)
#     return jnp.log(out[-1] / n_simu).sum()
