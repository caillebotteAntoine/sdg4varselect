"""
Module for BIC and eBIC computation.

Created by antoine.caillebotte@inrae.fr
"""

import jax.numpy as jnp
import scipy.special


def log_binom(n, k):
    """Computes log(n choose k) in a numerically stable way using log-gamma functions.

    Parameters
    ----------
    n : int
        Number of things.
    k : int
        Number of elements taken.

    Returns
    -------
    float
        The natural logarithm of the total number of combinations.
    """
    return (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k + 1)
        - scipy.special.gammaln(n - k + 1)
    )


def eBIC(  # pylint:disable = C0103
    theta_hd, log_likelihood, n, gamma=1.0
) -> jnp.ndarray:
    """Compute the extended Bayesian Information Criterion (eBIC) for a given model.

    Parameters
    ----------
    theta_hd : jnp.ndarray
        High-dimensional parameter vector, where non-zero values indicate estimated parameters.
    log_likelihood : jnp.ndarray
        Log-likelihood of the model, evaluated at the estimated parameters.
    n : int
        Sample size used to estimate the model.
    gamma: float
        Tuning parameter for the eBIC penalty term.

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
    ebic_pen = 2 * gamma * log_binom(theta_hd.shape[0], k)
    assert ebic_pen.shape == log_likelihood.shape

    return -2 * log_likelihood + k * jnp.log(n) + ebic_pen


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


def compute_metrics(x, x_star, tol=0):
    """
    Compute F1-score, support recovery rate, MSE, FDR, TNR, PPV, and TPR.

    Parameters
    ----------
    x : list of np.ndarray
        List of estimated coefficient vectors.
    x_star : np.ndarray
        True coefficient vector.
    tol : float, optional
        Tolerance for considering a coefficient as nonzero.

    Returns
    -------
    dict
        Dictionary containing the computed metrics:
        - "F1-score": Balance between precision and recall.
        - "Support Recovery Rate": Proportion of simulations recovering the exact support.
        - "MSE": Mean Squared Error between estimated and true coefficients.
        - "FDR": False Discovery Rate, proportion of selected variables that are false positives.
        - "TNR": True Negative Rate, proportion of correctly identified zero coefficients.
        - "PPV": Positive Predictive Value (Precision), proportion of selected variables that are true positives.
        - "TPR": True Positive Rate (Sensitivity), proportion of true variables correctly selected.
        - "ACC": Accuracy, proportion of true and false variables correctly selected.
    """
    if isinstance(x, list) or (isinstance(x, jnp.ndarray) and len(x.shape) > 1):
        metrics = [compute_metrics(xx, x_star, tol=tol) for xx in x]

        return {
            key: jnp.array([xx[key] for xx in metrics]) for key in metrics[0].keys()
        }

    metrics = {}
    true_support = jnp.abs(x_star) > tol
    estimated_support = jnp.abs(x) > tol

    P = jnp.sum(true_support)
    N = jnp.sum(~true_support)

    tp = jnp.sum(true_support & estimated_support)  # True Positives
    fp = jnp.sum(~true_support & estimated_support)  # False Positives
    fn = jnp.sum(true_support & ~estimated_support)  # False Negatives
    tn = jnp.sum(~true_support & ~estimated_support)  # True Negatives

    metrics["Accuracy"] = (tp + tn) / (P + N)  # Accuracy
    metrics["Precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0  #  PPV = Precision
    metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Sensitivity

    metrics["False Discovery Rate"] = fp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["True Negative Rate"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    ppv = metrics["Precision"]
    tpr = metrics["Sensitivity"]
    metrics["F1-score"] = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0

    metrics["Support Recovery Rate"] = int(
        jnp.array_equal(true_support, estimated_support)
    )
    metrics["MSE"] = jnp.mean((x - x_star) ** 2)

    metrics["TP+FP"] = tp + fp
    return metrics
