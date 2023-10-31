import sdg4varselect.gradient as sdg
import pytest
import jax.numpy as jnp

# pytest --cov-report term-missing --cov=sdg4varselect tests/


def test_prox():
    theta = jnp.array([-3, -2, -1, 1, 2, 3])

    print(sdg.prox(theta, 0.5, 4, 1))
    assert (sdg.prox(theta, 0.5, 4, 1) == jnp.array([-2, 0, 0, 0, 0, 2])).any()
