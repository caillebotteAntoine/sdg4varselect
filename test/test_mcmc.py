"""tests for the MCMC class"""

# pylint: disable=all
# import pytest
# import numpy as np

import jax.random as jrd
import jax.numpy as jnp

import parametrization_cookbook.jax as pc

from sdg4varselect._mcmc import MCMC, gibbs_sampler
from jax.scipy.stats import multivariate_normal

parametrization = pc.NamedTuple(
    mu1=pc.RealPositive(scale=0.5, shape=(1,)),
    gamma2_1=pc.MatrixDiagPosDef(scale=0.001, dim=1),
)

params0 = {"mu1": jnp.array([0.5]), "gamma2_1": jnp.array([[0.00025]])}
theta0_reals1d = parametrization.params_to_reals1d(**params0)


def likelihood_array(theta_reals1d, **kwargs):
    """return likelihood"""
    params = parametrization.reals1d_to_params(theta_reals1d)

    return multivariate_normal(
        jnp.array([kwargs["x"]]).T,
        params.mu1,
        params.gamma2_1,
    )


x_mcmc = MCMC(x0=1.0, sd=1, size=10, likelihood=likelihood_array, name="x")
x = x_mcmc.data


# def gibbs(key):
#     for i in range(400):
#         key = x_mcmc.gibbs_sampler_step(
#             key,
#             theta0_reals1d=theta0_reals1d,
#             x=x,
#         )


# gibbs_sampler(
#     jrd.PRNGKey(0),
#     "x",
#     0.1,
#     likelihood_array,
#     theta0_reals1d=theta0_reals1d,
#     x=x,
# )


# def test_gibbs_sampler():
#     x = 4 + 0.5 * jrd.normal(jrd.PRNGKey(0), (10,))

#     key_out, data, nacceptance = gibbs_sampler(
#         jrd.PRNGKey(0),
#         "x",
#         0.1,
#         likelihood_array,
#         theta0_reals1d,
#         x=x,
#     )

#     assert data.shape == x.shape

#     assert (
#         data
#         - np.array(
#             [4.451, 3.421, 3.575, 3.984, 5.116, 3.061, 3.506, 3.764, 4.167, 4.884]
#         )
#     ).sum() < 1e-3

#     assert nacceptance == 6


# def test_MCMC_repr():
#     x_mcmc = MCMC(1.0, sd=1, size=10, likelihood=likelihood_array, name="x")
#     acceptance_rate_first = x_mcmc.acceptance_rate()[0]

#     assert str(x_mcmc) == "[x]([1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]) [mean = 1.0, var = 0.0]"

#     x_mcmc.gibbs_sampler_step(jrd.PRNGKey(0), theta0_reals1d, x=x_mcmc.data)
#     x_mcmc.reset()

#     assert x_mcmc.sd == [1]
#     assert x_mcmc.acceptance_rate() == [acceptance_rate_first]
#     assert x_mcmc.lbd == 0.01
