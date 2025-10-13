# pylint: disable=all

import numpy as np
from sdg4varselect import _criterion_bic_ebic as crit

import jax.numpy as jnp


def test_log_binom_basic():
    # log_binom(5, 2) = log(10)
    result = crit.log_binom(5, 2)
    expected = np.log(10)
    assert np.isclose(result, expected, atol=1e-8)


def test_log_binom_edge_cases():
    # log_binom(n, 0) = 0, log_binom(n, n) = 0
    assert np.isclose(crit.log_binom(7, 0), 0)
    assert np.isclose(crit.log_binom(7, 7), 0)


def test_BIC_single():
    theta_hd = jnp.array([0.0, 1.0, 0.0, 2.0])
    log_likelihood = jnp.array(5.0)
    n = 100
    k = 2  # two nonzero
    expected = -2 * 5.0 + k * np.log(n)
    result = crit.BIC(theta_hd, log_likelihood, n)
    assert np.isclose(result, expected)


def test_eBIC_single():
    theta_hd = jnp.array([0.0, 1.0, 0.0, 2.0])
    log_likelihood = jnp.array(5.0)
    n = 100
    k = 2
    ebic_pen = 2 * crit.log_binom(4, k)
    expected = -2 * 5.0 + k * np.log(n) + ebic_pen
    result = crit.eBIC(theta_hd, log_likelihood, n)
    assert np.isclose(result, expected)


def test_BIC_shape_assertion():
    theta_hd = jnp.array([0.0, 1.0, 0.0])
    log_likelihood = jnp.array(3.0)
    n = 10
    # Should not raise
    crit.BIC(theta_hd, log_likelihood, n)


def test_eBIC_shape_assertion():
    theta_hd = jnp.array([0.0, 1.0, 0.0])
    log_likelihood = jnp.array(3.0)
    n = 10
    # Should not raise
    crit.eBIC(theta_hd, log_likelihood, n)


def test_compute_metrics_perfect_recovery():
    x_star = jnp.array([1.0, 0.0, 2.0, 0.0])
    x = jnp.array([1.0, 0.0, 2.0, 0.0])
    metrics = crit.compute_metrics(x, x_star)
    assert metrics["Support Recovery Rate"] == 1
    assert metrics["F1-score"] == 1
    assert metrics["Accuracy"] == 1
    assert metrics["MSE"] == 0


def test_compute_metrics_partial_recovery():
    x_star = jnp.array([1.0, 0.0, 2.0, 0.0])
    x = jnp.array([1.0, 0.0, 0.0, 0.0])
    metrics = crit.compute_metrics(x, x_star)
    assert metrics["Support Recovery Rate"] == 0
    assert metrics["F1-score"] < 1
    assert metrics["Accuracy"] < 1
    assert metrics["MSE"] > 0


def test_compute_metrics_list_input():
    x_star = jnp.array([1.0, 0.0, 2.0, 0.0])
    x_list = [jnp.array([1.0, 0.0, 2.0, 0.0]), jnp.array([0.0, 0.0, 2.0, 0.0])]
    metrics = crit.compute_metrics(x_list, x_star)
    assert isinstance(metrics["F1-score"], jnp.ndarray)
    assert metrics["Support Recovery Rate"].shape == (2,)


def test_compute_metrics_tol():
    x_star = jnp.array([1e-8, 0.0, 2.0, 0.0])
    x = jnp.array([0.0, 0.0, 2.0, 0.0])
    metrics = crit.compute_metrics(x, x_star, tol=1e-6)
    # The first entry should be considered zero
    assert metrics["Support Recovery Rate"] == 1


def test_compute_metrics_all_zeros():
    x_star = jnp.zeros(5)
    x = jnp.zeros(5)
    metrics = crit.compute_metrics(x, x_star)
    assert metrics["Support Recovery Rate"] == 1
    assert metrics["F1-score"] == 0
    assert metrics["Accuracy"] == 1
    assert metrics["MSE"] == 0
