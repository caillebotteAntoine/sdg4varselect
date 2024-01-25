# import sdg4varselect.linear_model as lm
# import pytest
# import numpy as np
# import jax.numpy as jnp


# def test_linear_curve_float():
#     x = np.array([0, 1, 2, 3, 4, 5, 6])
#     out = lm.linear_curve_float(
#         x,
#         intercept=3,
#         slope=2,
#     )

#     assert (out - np.array([3, 5, 7, 9, 11, 13, 15])).sum() < 1e-3


# def test_linear_curve():
#     N = 3
#     J = 5

#     time = [[j * i for j in range(J)] for i in range(N)]
#     data = [i + 1 for i in range(N)]

#     # Test matrix
#     out = lm.linear_curve(
#         time=jnp.array(time),
#         intercept=jnp.array(data),
#         slope=jnp.array(data),
#     )

#     assert out.shape == (N, J)

#     assert (
#         (out - jnp.array([[1, 1, 1, 1, 1], [2, 4, 6, 8, 10], [3, 9, 15, 21, 27]])) ** 2
#     ).sum() < 1e-3

#     # Test vector
#     out = lm.linear_curve(
#         time=jnp.array(time[0]),
#         intercept=jnp.array(data),
#         slope=jnp.array(data),
#     )
#     assert out.shape == (N, J)

#     assert (
#         (
#             out
#             - jnp.array(
#                 [
#                     [1, 1, 1, 1, 1],
#                     [2, 2, 2, 2, 2],
#                     [3, 3, 3, 3, 3],
#                 ]
#             )
#         )
#         ** 2
#     ).sum() < 1e-3


# def test_gaussian_prior():
#     assert (lm.gaussian_prior(4, 4, 9) + 0.5 * np.log(2 * np.pi * 9)) < 1e-3

#     assert lm.gaussian_prior(np.array([3, 4]), 4, 9).shape == (2,)

#     assert (lm.gaussian_prior(2, 1, 1) + (np.log(2 * np.pi) + 1) / 2).sum() < 1e-3
