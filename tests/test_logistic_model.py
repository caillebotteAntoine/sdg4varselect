import sdg4varselect.logistic_model as lm
import pytest
import numpy as np
import jax.numpy as jnp


def test_logistic_curve_float():
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    out = lm.logistic_curve_float(
        x,
        supremum=3,
        midpoint=3,
        growth_rate=2,
    )

    out == np.array([0.547, 0.807, 1.133, 1.5, 1.867, 2.193, 2.453])

    assert (
        out - np.array([0.547, 0.807, 1.133, 1.5, 1.867, 2.193, 2.453])
    ).sum() < 1e-3

    assert (
        lm.logistic_curve_float(
            1000,
            supremum=3,
            midpoint=3,
            growth_rate=2,
        )
        == 3
    )


def test_logistic_curve():
    N = 3
    J = 5

    time = [[j * i + 1 for j in range(J)] for i in range(N)]
    data = [i + 1 for i in range(N)]

    # Test matrix
    out = lm.logistic_curve(
        time=jnp.array(time),
        supremum=jnp.array(data),
        midpoint=jnp.array(data),
        growth_rate=jnp.array(data),
    )

    assert out.shape == (N, J)

    assert (
        (
            out
            - jnp.array(
                [
                    [0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.755, 1.0, 1.245, 1.462, 1.635],
                    [1.018, 1.5, 1.982, 2.374, 2.642],
                ]
            )
        )
        ** 2
    ).sum() < 1e-3

    # Test vector
    out = lm.logistic_curve(
        time=jnp.array(time[0]),
        supremum=jnp.array(data),
        midpoint=jnp.array(data),
        growth_rate=jnp.array(data),
    )
    assert out.shape == (N, J)

    assert (
        (
            out
            - jnp.array(
                [
                    [0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.755, 0.755, 0.755, 0.755, 0.755],
                    [1.018, 1.018, 1.018, 1.018, 1.018],
                ]
            )
        )
        ** 2
    ).sum() < 1e-3
