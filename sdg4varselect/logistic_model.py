import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit, grad

from sdg4varselect import gaussian_prior, logistic_curve
from sdg4varselect.miscellaneous import namedTheta


# ==== partial_loss ==== #
@jit
def loss_without_prior(theta, Y, time, phi1, phi2, phi3) -> jnp.ndarray:
    pred = logistic_curve(time, phi1, phi2, phi3)
    out = jnp.sum(jnp.power(Y - pred, 2))
    return jnp.sum(-out / (2 * theta.sigma2))


@jit
def loss_without_prior_array(theta, Y, time, phi1, phi2, phi3):
    out = [
        loss_without_prior(theta, Y[i], time, phi1[i], phi2[i], phi3[i])
        for i in range(len(phi1))
    ]
    return jnp.array(out)


# ==== Loss_array ==== #
@jit
def loss_array(theta, Y, time, phi1, phi2, phi3):
    latent_prior = (
        gaussian_prior(phi1, theta.beta1, theta.gamma2_1)
        + gaussian_prior(phi2, theta.beta2, theta.gamma2_2)
        + gaussian_prior(phi3, theta.beta3, 1)
    )

    return latent_prior + loss_without_prior_array(theta, Y, time, phi1, phi2, phi3)


jac_loss = jit(jacrev(loss_array))


@jit
def loss(theta, Y, time, phi1, phi2, phi3):
    out = loss_array(theta, Y, time, phi1, phi2, phi3)
    return out.mean()


grad_loss = jit(grad(loss))


# === just for compilation and test ===
def model(time, phi1, phi2, phi3, **kwargs):
    N = len(phi1)
    out = [logistic_curve(time, phi1[i], phi2[i], phi3[i]) for i in range(N)]
    return jnp.array(out)


if __name__ == "__main__":

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

    theta, thetaType = namedTheta(
        beta1=np.array([300.0]),
        gamma2_1=np.array([30.0]),
        beta2=np.array([400.0]),
        gamma2_2=np.array([30.0]),
        beta3=np.array([200.0]),
        sigma2=np.array([10.0]),
    )
