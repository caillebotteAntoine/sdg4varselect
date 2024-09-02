import pytest

# pylint: disable=C0116, W0621
# import numpy as np
import jax.numpy as jnp

from sdg4varselect.learning_rate import LearningRate

# pytest --cov-report term-missing --cov=sdg4varselect tests/


def test_learning_rate():
    x = LearningRate(1, 1.5, 3, 0.7, 3, 10)

    assert x.preheating == 1
    assert x.coef_preheating == 1.5
    assert x.heating == 2
    assert x.coef_heating == 0.7
    assert x.max == 3
    assert x.step_flat == 10

    x = LearningRate(1, 1.5, None, 0.7, 3, 10)
    assert jnp.isnan(x.heating)

    with pytest.raises(TypeError) as excinfo:
        LearningRate(1.0, 1.5, 3, 0.7, 3, 10)

    assert str(excinfo.value) == "preheating must be int"

    with pytest.raises(TypeError) as excinfo:
        LearningRate(1, "1.5", 3, 0.7, 3, 10)

    assert str(excinfo.value) == "coef_preheating must be int or float"

    with pytest.raises(TypeError) as excinfo:
        LearningRate(1, 1.5, 3.0, 0.7, 3, 10)

    assert str(excinfo.value) == "heating must be int"

    with pytest.raises(TypeError) as excinfo:
        LearningRate(1, 1.5, 3, "0.7", 3, 10)

    assert str(excinfo.value) == "coef_heating must be int or float"

    with pytest.raises(TypeError) as excinfo:
        LearningRate(1, 1.5, 3, 0.7, "3", 10)

    assert str(excinfo.value) == "value_max must be int or float"

    with pytest.raises(TypeError) as excinfo:
        LearningRate(1, 1.5, 3, 0.7, 3, "10")

    assert str(excinfo.value) == "step_flat must be int or float"


def test_learning_static():
    x = LearningRate.zero()
    assert x.preheating == 1000
    assert x.max == 0

    x = LearningRate.one()
    assert x.preheating == 0
    assert x.max == 1

    x = LearningRate.from_0_to_1(10, -2)
    assert x.preheating == 10
    assert x.coef_preheating == -2

    x = LearningRate.from_1_to_0(10, -2)
    assert x.heating == 9
    assert x.coef_heating == -2


# def test_learning_call():
#     f = LearningRate(10, -2, 20, 0.75, step_flat=2)

#     assert f(10) == 1
#     assert f(1) == 0
#     assert f(3) == 0.2465969639416065
#     assert f(25) == 0.26084743001221455


def test_learning_repr():
    f = LearningRate(10, -2, 20, 0.75)

    assert (
        str(f)
        == "LearningRate :\n\t i ->\t | exp(-2*(1-i/10))\t if i < 10\n\t\t | ( i - 0.75)^-0.75\t if i >= 19\n\t\t | 1\t otherwise"
    )
