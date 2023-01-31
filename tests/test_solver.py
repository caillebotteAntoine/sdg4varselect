import numpy as np
import pytest

import sdg4varselect as sdg

import jax.numpy as jnp
from jax import jit, value_and_grad

N = 100
Z1 = np.random.normal(5, 2, size=N)

print(np.var(Z1))


@jit
def f(Z1, **kwargs):
    return 3 * Z1 + 14


Y = f(Z1) + np.random.normal(0, 0.5, size=N)


def loglikelihood(i, theta, Y, Z1, **kwargs):
    out = np.sum(np.power(Y[i] - f(Z1[i]), 2))
    return -out / (2 * theta.sigma2)


@jit
def loss(theta, Y, Z1, **kwargs):
    out = jnp.mean(jnp.power(Y - f(Z1), 2))
    return -out / (2 * theta.sigma2[0])


s = sdg.solver_init(
    sdg.solver(),
    theta0={"mu1": 10, "omega2_1": 1},
    mean_name="mu",
    variance_name="omega2_",
    mcmc_name="Z",
    dim={"Z1": 100},
    sd={"Z1": 0.5},
)

s.add_variable("Y", Y)
s.add_parameter(sdg.parameter.par_noise_variance(1, "Y", f, "sigma2"))
s.set_data("Y", "Z1")

print(s)

s.init_parameters()
s.step_size(sdg.burnin_fct.from_1_to_0(200, 0.75))
s.SGD(250, loglikelihood, jit(value_and_grad(loss)))

print(s)

mu = s.theta().mu1
omega2 = s.theta().omega2_1

assert np.abs(mu / 5 - 1) < 0.05

s.to_csv("../cout.txt")
