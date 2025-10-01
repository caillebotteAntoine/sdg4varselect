"""
List of the main modules of the package.
"""

from jax import config

# Algorithm
from .algo import *

# Models
from .models import *


# MCMC
from ._mcmc import MCMC
from ._chain import Chain

# Outputs
from . import outputs

from .outputs import (
    Sdg4vsResults,
    GDResults,
    SGDResults,
    MultiGDResults,
    RegularizationPath,
)

# Exceptions
from .exceptions import *

config.update("jax_enable_x64", True)
