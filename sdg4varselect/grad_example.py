import jax.numpy as jnp
from jax import jit, grad

from burnin_fct import burnin_fct
from miscellaneous import time_profiler

import numpy as np

# === Data simulation === #
from data_sim import loglikelihood
from data_sim import Y_obs, time_obs, phi1_obs, phi2_obs, phi3_obs
from data_sim import s, theta_star
from math import pi


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


# print(gaussian_prior(phi1_obs, np.array([np.mean(phi1_obs)]), np.var(phi1_obs)))


@jit
def model(time, phi1, phi2, phi3, **kwargs):
    N = len(phi1)
    out = [jit_logistic(time, phi1[i], phi2[i], phi3[i]) for i in range(N)]
    return jnp.array(out)


# print(model(time_obs, phi1_obs, phi2_obs, phi3_obs))


@jit
def loss(theta, Y, time, phi1, phi2, phi3):
    N = len(phi1)
    latent_prior = (
        gaussian_prior(phi1, theta.beta1, theta.gamma2_1)
        + gaussian_prior(phi2, theta.beta2, theta.gamma2_2)
        + gaussian_prior(phi3, theta.beta3, 1)
    )

    pred = model(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2))
    out = latent_prior - out / (2 * theta.sigma2[0])

    return out / N  # - Loglikelihood for the maximization


# print(loss(theta_star, Y_obs, time_obs, phi1_obs, phi2_obs, phi3_obs))


def gradient_descent(loss_grad, theta0, step_size: float, **kwargs):
    eval, grad_eval = loss_grad(theta0, **kwargs)

    print("L = " + str(eval))
    print("grad = " + str(grad_eval))
    print("step_size = " + str(step_size))
    print("theta = " + str(theta0))
    print("\n\n")

    for par in theta0:
        theta0[par] += step_size * grad_eval[par]

    return theta0


grad_loss = jit(grad(loss))
step_size = burnin_fct(500, -4, 550, 0.75)

theta0 = {
    "beta1": np.array([300.0]),
    "gamma2_1": np.array([30.0]),
    "beta2": np.array([400.0]),
    "gamma2_2": np.array([30.0]),
    "beta3": np.array([100.0]),
    "sigma2": np.array([10.0]),
}


@time_profiler(nrun=1)
def GD(niter):
    for iter in range(niter):
        gradient_descent(
            grad_loss,
            theta0,
            step_size(iter),
            Y=Y_obs,
            time=time_obs,
            phi1=phi1_obs,
            phi2=phi2_obs,
            phi3=phi3_obs,
        )
    return theta0


# print(GD(5))

print(s)
s.step_size(step_size)
s.SGD(600, loglikelihood, grad_loss)
print(s)

s.to_csv("../cout.txt")
