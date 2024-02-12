"""
Create by antoine.caillebotte@inrae.fr
"""

from jax.config import config

# from .learning_rate import LearningRate
from ._MCMC import MCMC_chain
from ._regularization_function import eBIC, BIC, regularization_path

# import jax.numpy as jnp
# jnp.set_printoptions(threshold=200)  # jnp.inf)

# import jax.random as jrd
# from jax import jit, jacrev, jacfwd

config.update("jax_enable_x64", True)
