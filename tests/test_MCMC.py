from sdg4varselect.MCMC import MCMC_chain
import pytest
import numpy as np

from sdg4varselect import jrd
from sdg4varselect.logistic_model import gaussian_prior
import parametrization_cookbook.jax as pc

parametrization = pc.NamedTuple(
    mu1=pc.RealPositive(scale=0.5),
    gamma2_1=pc.RealPositive(scale=0.001),
)

params0 = {"mu1": 0.5, "gamma2_1": 0.00025}
theta0_reals1d = parametrization.params_to_reals1d(**params0)


def likelihood_array(theta_reals1d, parametrization, **kwargs):
    """return likelihood"""
    params = parametrization.reals1d_to_params(theta_reals1d)

    return gaussian_prior(
        kwargs["x"],
        params.mu1,
        params.gamma2_1,
    )


x = np.random.normal(4, 0.5, 10)
x_mcmc = MCMC_chain(1.0, sd=1, size=10, likelihood=likelihood_array, name="x")


def gibbs(key):
    for i in range(400):
        key = x_mcmc.gibbs_sampler_step(
            key,
            theta0_reals1d,
            parametrization=parametrization,
            x=x,
        )


gibbs(jrd.PRNGKey(0))
