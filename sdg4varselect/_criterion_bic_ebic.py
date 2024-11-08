"""
Module for BIC and eBIC computation.

Created by antoine.caillebotte@inrae.fr
"""

import jax.numpy as jnp
import scipy.special


def eBIC(theta_hd, log_likelihood, n) -> jnp.ndarray:  # pylint:disable = C0103
    """Compute the extended Bayesian Information Criterion (eBIC) for a given model.

    Parameters
    ----------
    theta_hd : jnp.ndarray
        High-dimensional parameter vector, where non-zero values indicate estimated parameters.
    log_likelihood : jnp.ndarray
        Log-likelihood of the model, evaluated at the estimated parameters.
    n : int
        Sample size used to estimate the model.

    Returns
    -------
    jnp.ndarray
        The computed eBIC value for the model.

    Notes
    -----
    eBIC = k*ln(n) - 2*ln(L) + 2*ln(C^k_p)

    where :
        - k is the number of parameter estimated (ie non zero parameter in HD parameter)
        - n is the sample size
        - L the maximzed value of the likelihood function
    """
    assert len(theta_hd.shape) == 1

    k = (theta_hd != 0).sum()
    assert k.shape == log_likelihood.shape
    ebic_pen = scipy.special.binom(theta_hd.shape[0], k)
    assert ebic_pen.shape == log_likelihood.shape

    return -2 * log_likelihood + k * jnp.log(n) + 2 * jnp.log(ebic_pen)


def BIC(theta_hd, log_likelihood, n) -> jnp.ndarray:  # pylint:disable = C0103
    """Compute the Bayesian Information Criterion (BIC) for a given model.

    Parameters
    ----------
    theta_hd : jnp.ndarray
        High-dimensional parameter vector, where non-zero values indicate estimated parameters.
    log_likelihood : jnp.ndarray
        Log-likelihood of the model, evaluated at the estimated parameters.
    n : int
        Sample size used to estimate the model.

    Returns
    -------
    jnp.ndarray
        The computed BIC value for the model.

    Notes
    -----
    BIC = k*ln(n) - 2*ln(L)
    where:
        - k is the number of parameter estimated (ie non zero parameter in HD parameter)
        - n is the sample size
        - L the maximzed value of the likelihood function
    """
    assert len(theta_hd.shape) == 1

    k = (theta_hd != 0).sum()
    assert k.shape == log_likelihood.shape

    return -2 * log_likelihood + k * jnp.log(n)
