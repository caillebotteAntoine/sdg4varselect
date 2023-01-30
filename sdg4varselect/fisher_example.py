from math import pi

import jax.numpy as jnp
import numpy as np
import parametrization_cookbook.jax as pc
from burnin_fct import burnin_fct

# === Data simulation === #
from data_sim import (
    N,
    Y_obs,
    loglikelihood,
    phi1_obs,
    phi2_obs,
    phi3_obs,
    s,
    theta_star,
    time_obs,
)
from jax import grad, jit
from miscellaneous import time_profiler
from parameter import par_grad, par_grad_ind

parametrization = s.parametrization(
    beta1=pc.Real(shape=1),
    beta2=pc.Real(shape=1),
    beta3=pc.Real(shape=1, scale=1.5),
    gamma2_1=pc.Real(shape=1),
    gamma2_2=pc.Real(shape=1),
    sigma2=pc.Real(shape=1),
)

theta0 = {
    "beta1": np.array([300.0]),
    "gamma2_1": np.array([30.0]),
    "beta2": np.array([400.0]),
    "gamma2_2": np.array([30.0]),
    "beta3": np.array([100.0]),
    "sigma2": np.array([10.0]),
}


@jit
def jit_logistic(
    x, supremum: float, midpoint: float, growth_rate: float
) -> jnp.ndarray:
    return supremum / (1 + jnp.exp(-(x - midpoint) / growth_rate))


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    # Computation of the current target distrubtion score
    out = jnp.log(2 * pi * variance) + jnp.power(data - mean, 2) / variance
    return -jnp.sum(out) / 2


@jit
def model(time, phi1, phi2, phi3, **kwargs):
    N = len(phi1)
    out = [jit_logistic(time, phi1[i], phi2[i], phi3[i]) for i in range(N)]
    return jnp.array(out)


@jit
def loss_id(theta, i, Y, time, phi1, phi2, phi3):
    latent_prior = (
        gaussian_prior(phi1[i], theta.beta1, theta.gamma2_1)
        + gaussian_prior(phi2[i], theta.beta2, theta.gamma2_2)
        + gaussian_prior(phi3[i], theta.beta3, 1)
    )

    pred = model(time, phi1[i], phi2[i], phi3[i])
    out = jnp.sum(jnp.power(Y[i] - pred, 2))
    out = latent_prior - out / (2 * theta.sigma2[0])

    return out


def fisher(grad):
    if not isinstance(grad, list):
        g = np.array([np.concatenate([grad[k] for k in grad])])  # vector to matrix
        return np.dot(g.T, g)
    else:
        fim = [fisher(grad[i]) for i in range(len(grad))]
        out = fim[0]
        for i in range(1, len(fim)):
            out += fim[i]
        return out


def get_grad_ind(loss_grad, theta: dict[str, np.ndarray], Y, time, phi1, phi2, phi3):
    N = len(phi1)

    gradient_list = []
    gradient = theta.copy()
    for v in theta:
        gradient[v] = np.array([0])

    for i in range(N):
        gradient_list.append(loss_grad(theta, Y[i], time, phi1[i], phi2[i], phi3[i]))
        for v in theta:
            gradient[v] += gradient_list[-1][v]

    for v in theta:
        gradient[v] /= N
    return gradient_list, gradient


grad_loss = grad(loss_id)
step_size = burnin_fct(500, -4, 550, 0.75)


print(s)
s.step_size(step_size)

dim = np.sum([len(x) for x in theta0.values()])
gradients = []
for i in range(N):
    gradients.append(par_grad_ind(dim, i, grad_loss))

gradient = par_grad(theta0, gradients, "fisher")

s.SGD2(0, loglikelihood, gradient)


@time_profiler
def test_grad_with_par():
    gradient.compute_all_individual_grad(s)


@time_profiler
def test_grad():
    for i in range(N):
        grad_loss(theta0, i, Y_obs, time_obs, phi1_obs, phi2_obs, phi3_obs)


def loss(theta, Y, time, phi1, phi2, phi3):
    N = len(phi1)
    latent_prior = (
        gaussian_prior(phi1, theta["beta1"], theta["gamma2_1"])
        + gaussian_prior(phi2, theta["beta2"], theta["gamma2_2"])
        + gaussian_prior(phi3, theta["beta3"], 1)
    )

    pred = model(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2))
    out = latent_prior - out / (2 * theta["sigma2"][0])

    return out / N  # - Loglikelihood for the maximization


full_grad = grad(loss)


@time_profiler
def test_full_grad():
    full_grad(theta0, Y_obs, time_obs, phi1_obs, phi2_obs, phi3_obs)


test_grad_with_par()
test_grad()
test_full_grad()

print(s)

s.to_csv("../cout.txt")
