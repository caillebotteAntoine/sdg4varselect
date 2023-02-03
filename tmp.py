import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from sdg4varselect.miscellaneous import time_profiler


# ========================================================#
@jit
def f(args):
    (x, y) = args
    return (x - 1) ** 2 + (y + 10) ** 2


@time_profiler
def DG(f, niter, eps=1e-1):
    f_grad = grad(f)
    x0 = jnp.array([-1.0, 2.0])
    for i in range(niter):
        x0 -= eps * f_grad(x0)

    return x0


# ========================================================#
@jit
def g(x, y):
    return (x - 1) ** 2 + (y + 10) ** 2


@time_profiler
def DG_2dim(g, niter, eps=1e-1):
    g_grad_x = grad(g, 0)
    g_grad_y = grad(g, 1)

    x0 = jnp.array([-1.0, 2.0])
    for i in range(niter):
        x0 -= eps * jnp.array([g_grad_x(*x0), g_grad_y(*x0)])

    return x0


# ========================================================#
def h(args):
    x = args["x"]
    y = args["y"]

    out = (x - 1) ** 2 + (y + 10) ** 2
    return out[0]


@jit
def h_jit(args):
    x = args["x"]
    y = args["y"]

    out = (x - 1) ** 2 + (y + 10) ** 2
    return out[0]


@time_profiler
def DG_dict(h, niter, eps=1e-1):

    h_grad = grad(h)
    x0 = {"x": np.array([-1.0]), "y": np.array([2.5])}
    for i in range(niter):
        grad_eval = h_grad(x0)
        for par in x0:
            x0[par] = x0[par] - eps * grad_eval[par]

    return x0


# ========================================================#

niter = 10
print(DG(f, niter))
print(DG_2dim(g, niter))
print(DG_dict(h, niter))
print(DG_dict(h_jit, niter))
# for 1000 loops
# 10 loops:    2.001 s per loop
# 10 loops:    5.169 s per loop
# 7 loops:    9.449 s per loop
# 10 loops:    2.510 s per loop


class test:
    def __init__(self, x, func):
        self.__x = x

        self.__loglike = lambda *args, **kwargs: self.__x * func(*args, **kwargs)

    def val(self, y):
        return self.__loglike(y)


x = test(10, lambda x: x**2)

print(x.val(2))
