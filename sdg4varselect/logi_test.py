# Create by antoine.caillebotte@inrae.fr

import parametrization_cookbook.jax as pc
import numpy as np


import jax.numpy as jnp
import jax.random as jrd
from jax import jit, jacfwd
import functools

from collections import namedtuple


from sdg4varselect.data_handler import Data_handler
from sdg4varselect.Joint_model import (
    mem_simulation,
    cov_simulation,
    cox_simulation,
    gaussian_prior,
)


# =============================================== #
# ====================== PK ===================== #
# =============================================== #
@jit
def pk_curve_float(t, D: float, ka: float, Cl: float, V: float):
    return D * ka / (V * ka - Cl) * (jnp.exp(-ka * t) - jnp.exp(-Cl / V * t))


@jit
def pk_curve(
    time: jnp.ndarray,  # shape = (N,J)
    D: jnp.ndarray,  # shape = (1,) [:,None]
    ka: jnp.ndarray,  # shape = (N,) [:,None]
    Cl: jnp.ndarray,  # shape = (N,) [:,None]
    V: jnp.ndarray,  # shape = (N,) [:,None]
) -> jnp.ndarray:  # shape = (N,J)
    return (
        D
        * ka[:, None]
        / (V[:, None] * (ka[:, None] - Cl[:, None] / V[:, None]))
        * (jnp.exp(-ka[:, None] * time) - jnp.exp(-Cl[:, None] / V[:, None] * time))
    )
