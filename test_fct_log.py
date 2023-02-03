import jax.numpy as jnp
import numpy as np
from jax import grad, jacfwd, jacrev, jit
from jax import value_and_grad as vgrad

from sdg4varselect import gaussian_prior, logistic_curve
from sdg4varselect.miscellaneous import difftime, namedTheta


@jit
def model(time, phi1, phi2, phi3, **kwargs) -> jnp.ndarray:
    N = len(phi1)
    out = [logistic_curve(time, phi1[i], phi2[i], phi3[i]) for i in range(N)]
    return jnp.array(out)


# ==== Data simulation ==== #
N, J = 500, 200

eps = np.random.normal(0, np.sqrt(100), (N, J))
sim = {
    "time": np.linspace(100, 1500, num=J),
    "phi1": np.random.normal(200, np.sqrt(40), N),
    "phi2": np.random.normal(500, np.sqrt(100), N),
    "phi3": np.array([150 for i in range(N)]),
}

sim["Y"] = model(**sim) + eps

theta = namedTheta(
    beta1=np.array([300.0]),
    gamma2_1=np.array([30.0]),
    beta2=np.array([400.0]),
    gamma2_2=np.array([30.0]),
    beta3=np.array([200.0]),
    sigma2=np.array([10.0]),
)

nrun = 2000


# ==== partial_loglike ==== #
@jit
def partial_loglike_original(i: int, theta, Y, time, phi1, phi2, phi3) -> jnp.ndarray:
    pred = logistic_curve(time, phi1[i], phi2[i], phi3[i])
    out = jnp.sum(pow(Y[i] - pred, 2))
    return jnp.sum(-out / (2 * theta.sigma2))


@jit
def partial_loglike_one_indiv(theta, Y, time, phi1, phi2, phi3) -> jnp.ndarray:
    pred = logistic_curve(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2))
    return jnp.sum(-out / (2 * theta.sigma2))


# ==== Loss_array ==== #
@jit
def partial_loss_array_from_original(theta, Y, time, phi1, phi2, phi3):
    out = [
        partial_loglike_original(i, theta, Y, time, phi1, phi2, phi3)
        for i in range(len(phi1))
    ]
    return jnp.array(out)


@jit
def partial_loss_array_from_one_indiv(theta, Y, time, phi1, phi2, phi3):
    out = [
        partial_loglike_one_indiv(theta, Y[i], time, phi1[i], phi2[i], phi3[i])
        for i in range(len(phi1))
    ]
    return jnp.array(out)


@jit
def partial_loss_array_original(theta, Y, time, phi1, phi2, phi3):
    pred = model(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2), axis=1)
    out = -out / (2 * theta.sigma2[0])

    return out


print("")
difftime(
    partial_loss_array_from_original,
    partial_loss_array_from_one_indiv,
    partial_loss_array_original,
    nloop=2000,
    nrun=nrun,
)(theta, **sim)


partial_loss_array = partial_loss_array_from_one_indiv


@jit
def loss_array(theta, Y, time, phi1, phi2, phi3):

    latent_prior = (
        gaussian_prior(phi1, theta.beta1, theta.gamma2_1)
        + gaussian_prior(phi2, theta.beta2, theta.gamma2_2)
        + gaussian_prior(phi3, theta.beta3, 1)
    )

    return latent_prior + partial_loss_array(theta, Y, time, phi1, phi2, phi3)


# ==== Loss for MCMC ==== #
@jit
def loss_one_mcmc(theta, Y, time, phi1, phi2, phi3):
    latent_prior = gaussian_prior(phi1, theta.beta1, theta.gamma2_1)

    return latent_prior + partial_loss_array(theta, Y, time, phi1, phi2, phi3)


print("")
difftime(
    loss_array,
    loss_one_mcmc,
    nloop=2000,
    nrun=nrun,
)(theta, **sim)

# ==== Loss ==== #


@jit
def loss_from_array(theta, Y, time, phi1, phi2, phi3):
    out = loss_array(theta, Y, time, phi1, phi2, phi3)
    return jnp.mean(out)


@jit
def loss_original(theta, Y, time, phi1, phi2, phi3):
    latent_prior = (
        gaussian_prior(phi1, theta.beta1, theta.gamma2_1)
        + gaussian_prior(phi2, theta.beta2, theta.gamma2_2)
        + gaussian_prior(phi3, theta.beta3, 1)
    )

    pred = model(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2), axis=1)
    out = latent_prior - out / (2 * theta.sigma2[0])

    return jnp.mean(out)


print("")
difftime(
    loss_original,
    loss_from_array,
    nloop=2000,
    nrun=nrun,
)(theta, **sim)

loss = loss_from_array
loss
# ==== grad ==== #

vgrad_loss = jit(vgrad(loss))
grad_loss = jit(grad(loss))

print("")
difftime(
    grad_loss,
    vgrad_loss,
    nloop=2000,
    nrun=nrun,
)(theta, **sim)


jacrev_loss = jit(jacrev(loss))
jacfwd_loss = jit(jacfwd(loss))

print("")
difftime(
    jacrev_loss,
    jacfwd_loss,
    nloop=2000,
    nrun=200,
)(theta, **sim)
