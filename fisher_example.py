import numpy as np
from jax import jit, value_and_grad
import parametrization_cookbook.jax as pc

import sdg4varselect as sdg
from sdg4varselect.parameter import par_grad, par_grad_ind

# === Data simulation === #
from data_sim import get_data
from sdg4varselect.logistic_model import loss_id, loss, partial_loglikelihood
from sdg4varselect.miscellaneous import time_profiler, difftime

s = get_data(
    theta0={
        "beta1": np.array([300.0]),
        "gamma2_1": np.array([30.0]),
        "beta2": np.array([400.0]),
        "gamma2_2": np.array([30.0]),
        "beta3": np.array([200.0]),
        "sigma2": np.array([10.0]),
    }
)


parametrization = s.parametrization(
    beta1=pc.Real(shape=1),
    beta2=pc.Real(shape=1),
    beta3=pc.Real(shape=1, scale=1.5),
    gamma2_1=pc.Real(shape=1),
    gamma2_2=pc.Real(shape=1),
    sigma2=pc.Real(shape=1),
)


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


grad_loss = value_and_grad(loss_id)
step_size = sdg.burnin_fct(500, -4, 550, 0.75)


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


full_grad = value_and_grad(loss)


@time_profiler
def test_full_grad():
    full_grad(theta0, Y_obs, time_obs, phi1_obs, phi2_obs, phi3_obs)


test_grad_with_par()
test_grad()
test_full_grad()

print(s)

s.to_csv("../cout.txt")
