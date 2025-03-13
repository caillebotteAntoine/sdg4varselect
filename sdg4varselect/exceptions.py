"""
Define custom exceptions.

Create by antoine.caillebotte@inrae.fr
"""

import jax.numpy as jnp
import numpy as np


class Sdg4vsException(Exception):
    """Base of all Sdg4Varselect exceptions."""


class Sdg4vsNanError(Sdg4vsException):
    """Sdg4Varselect Error related to Nan in array."""


class Sdg4vsInfError(Sdg4vsException):
    """Sdg4Varselect Error related to Inf in array."""


class Sdg4vsWrongParametrization(Sdg4vsException):
    """Sdg4Varselect Error related to wrong parametrization in models."""


def _contains_nan_or_inf(x):
    """Checks if a dictionary contains NaN or Inf values in its entries.

    Supports scalars, JAX arrays, and lists of mixed types.

    Parameters
    ----------
    X : dict
        Dictionary where values can be int, float, jax.numpy.ndarray, or nested lists.

    Returns
    -------
    bool
        True if any NaN or Inf is found, False otherwise.
    """

    def check_value(
        value,
    ) -> (
        bool
    ):  # pylint: disable=missing-return-doc, missing-raises-doc,missing-any-param-doc
        """Checks a single value for NaN or Inf."""
        if isinstance(
            value, (int, bool, str)
        ):  # Integers and booleans are always finite
            return False
        if isinstance(value, float):  # Check directly for floats
            return jnp.isnan(value) or jnp.isinf(value)
        if isinstance(value, (jnp.ndarray, np.ndarray)):  # Check for JAX arrays
            return jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value))
        if isinstance(value, (list, tuple)):  # Recursively check lists/tuples
            return any(check_value(v) for v in value)
        if isinstance(value, dict):  # Recursively check lists/tuples
            return any(check_value(v) for v in value.values())

        raise Sdg4vsException(
            f"Unsupported type: {type(value)} in nan or inf detection"
        )  # Handle unexpected types

    return any(check_value(xx) for xx in x.values())
