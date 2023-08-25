from .learning_rate import learning_rate
from .MCMC import MCMC_chain
from .gradient import Gradient

import jax
import jax.random as jrd
import jax.numpy as jnp

jnp.set_printoptions(threshold=200)  # jnp.inf)

import jax.numpy as jnp
import jax.random as jrd
from jax import jit, jacrev


def print_array(x):
    out = ""
    for i in range(len(x)):
        if len(x[i].shape) != 0:
            out += "[ "
            for j in range(len(x[i])):
                out += "{:.3e}".format(x[i][j])

                if j != len(x[i]):
                    out += ", "
            out += "]"
        else:
            out += "{:.3e}".format(x[i])

        if i != len(x):
            out += ","

        if len(x[i].shape) != 0:
            out += "\n"

    print(out)
