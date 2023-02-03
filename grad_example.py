import jax.numpy as jnp
import parametrization_cookbook.jax as pc
import numpy as np
from sdg4varselect.burnin_fct import burnin_fct

# === Data simulation === #
from data_sim import get_data
from sdg4varselect.algorithm import algorithm
from sdg4varselect.miscellaneous import time_profiler
from sdg4varselect.logistic_model import (
    loss_without_prior_array,
    jac_loss,
    grad_loss,
)


s, thetaType = get_data(
    algorithm(),
    theta0={
        "beta1": np.array([300.0]),
        "gamma2_1": np.array([30.0]),
        "beta2": np.array([400.0]),
        "gamma2_2": np.array([30.0]),
        "beta3": np.array([100.0]),
        "sigma2": np.array([10.0]),
    },
)

parametrization = s.parametrization(
    beta1=pc.Real(shape=1),
    beta2=pc.Real(shape=1),
    beta3=pc.Real(shape=1),
    gamma2_1=pc.Real(shape=1),
    gamma2_2=pc.Real(shape=1),
    sigma2=pc.Real(shape=1),
)

"""
def grad_loss(theta, Y, time, phi1, phi2, phi3):
    jac = jacrev_loss(theta, Y, time, phi1, phi2, phi3)
    jac = jnp.array(jac).T[0]
    # fisher jac.T @ jac
    out = thetaType(*jac.mean(axis=0))
    return out
"""


def gradient_descent(loss_grad, theta0, step_size: float, **kwargs):
    eval, grad_eval = loss_grad(theta0, **kwargs)

    # print("L = " + str(eval))
    # print("grad = " + str(grad_eval))
    # print("step_size = " + str(step_size))
    # print("theta = " + str(theta0))
    # print("\n\n")

    for i in range(len(theta0)):
        theta0[i][0] += step_size * grad_eval[i]

    return theta0


step_size = burnin_fct(600, -4, 650, 0.75)

print(loss_without_prior_array(s.theta_to_params(), **s.data()))
print(grad_loss(s.theta_to_params(), **s.data()))
s.SGD(1, loss_without_prior_array, grad_loss)
s.reset_solver()

s.step_size(step_size)
# s.Heating(100, loss_without_prior_array)
s.SGD(600, loss_without_prior_array, grad_loss)
print(s)

s.to_csv("cout.txt")
