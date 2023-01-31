import numpy as np
from sdg4varselect.burnin_fct import burnin_fct

# === Data simulation === #
from data_sim import get_data
from jax import value_and_grad, jit
from sdg4varselect.miscellaneous import time_profiler
from sdg4varselect.logistic_model import loss, partial_loglikelihood


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


grad_loss = jit(value_and_grad(loss))
step_size = burnin_fct(500, -4, 550, 0.75)


@time_profiler(nrun=1)
def GD(niter):
    theta = s.theta()
    for iter in range(niter):
        gradient_descent(grad_loss, theta, step_size(iter), **s.data())
    return theta


# print(GD(5))

s.step_size(step_size)
s.SGD(600, partial_loglikelihood, grad_loss)
print(s)

s.to_csv("../cout.txt")
