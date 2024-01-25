# from sdg4varselect.learning_rate import learning_rate
# import pytest
# import numpy as np

# # pytest --cov-report term-missing --cov=sdg4varselect tests/


# def test_learning_rate():
#     x = learning_rate(1, 1.5, 3, 0.7, 3, 10)

#     assert x.step_heat == 1
#     assert x.coef_heat == 1.5
#     assert x.step_burnin == 2
#     assert x.coef_burnin == 0.7
#     assert x.scale == 3
#     assert x.step_flat == 10

#     x = learning_rate(1, 1.5, None, 0.7, 3, 10)
#     assert x.step_burnin is None

#     with pytest.raises(TypeError) as excinfo:
#         learning_rate(1.0, 1.5, 3, 0.7, 3, 10)

#     assert str(excinfo.value) == "step_heat must be int"

#     with pytest.raises(TypeError) as excinfo:
#         learning_rate(1, "1.5", 3, 0.7, 3, 10)

#     assert str(excinfo.value) == "coef_heat must be int or float"

#     with pytest.raises(TypeError) as excinfo:
#         learning_rate(1, 1.5, 3.0, 0.7, 3, 10)

#     assert str(excinfo.value) == "step_burnin must be int"

#     with pytest.raises(TypeError) as excinfo:
#         learning_rate(1, 1.5, 3, "0.7", 3, 10)

#     assert str(excinfo.value) == "coef_burnin must be int or float"

#     with pytest.raises(TypeError) as excinfo:
#         learning_rate(1, 1.5, 3, 0.7, "3", 10)

#     assert str(excinfo.value) == "scale must be int or float"

#     with pytest.raises(TypeError) as excinfo:
#         learning_rate(1, 1.5, 3, 0.7, 3, "10")

#     assert str(excinfo.value) == "step_flat must be int or float"


# def test_learning_static():
#     x = learning_rate.zero()
#     assert x.step_heat == 1000
#     assert x.scale == 0

#     x = learning_rate.one()
#     assert x.step_heat == 0
#     assert x.scale == 1

#     x = learning_rate.from_0_to_1(10, -2)
#     assert x.step_heat == 10
#     assert x.coef_heat == -2

#     x = learning_rate.from_1_to_0(10, -2)
#     assert x.step_burnin == 9
#     assert x.coef_burnin == -2


# def test_learning_call():
#     f = learning_rate(10, -2, 20, 0.75, step_flat=2)

#     assert f(10) == 1
#     assert f(1) == 0
#     assert f(3) == 0.2465969639416065
#     assert f(25) == 0.26084743001221455


# def test_learning_repr():
#     f = learning_rate(10, -2, 20, 0.75)

#     assert (
#         str(f)
#         == "learning_rate :\n\t i ->\t | exp(-2*(1-i/10))\t if i < 10\n\t\t | ( i - 0.75)^-0.75\t if i >= 19\n\t\t | 1\t otherwise"
#     )
