"""
Module for Gradient descent algorithm preconditioned by the fisher information matrix.

Create by antoine.caillebotte@inrae.fr
"""

import itertools
from collections import namedtuple

import jax.numpy as jnp

from sdg4varselect.learning_rate import create_multi_step_size, LearningRate
from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.algo.abstract.abstract_algo_fit import AbstractAlgoFit
from sdg4varselect.outputs import GDResults

from sdg4varselect.algo.preconditioner import (
    # Fisher,
    # AdaGrad,
    AbstractPreconditioner,
)


# GradientDescentFIMSettings = namedtuple(
#     "GradientDescentFIMSettings",
#     ("step_size_grad", "step_size_approx_sto", "step_size_fisher"),
# )


# def get_gdfim_settings(preheating, heating, learning_rate=1e-8):
#     """return 3 step size well parametred"""
#     return GradientDescentFIMSettings(
#         step_size_grad={
#             "learning_rate": learning_rate,
#             "preheating": preheating,
#             "heating": heating,
#             "max": 0.9,
#         },
#         step_size_approx_sto={
#             "learning_rate": learning_rate,
#             "preheating": preheating,
#             "heating": None,
#             "max": 1,
#         },
#         step_size_fisher={
#             "learning_rate": learning_rate,
#             "preheating": preheating,
#             "heating": None,
#             "max": 0.9,
#         },
#     )


class GradientDescentFIM(AbstractAlgoFit):
    """Gradient descent algorithm preconditioned by the fisher information matrix"""

    def __init__(
        self,
        max_iter: int,
        step_size: LearningRate,
        preconditioner: AbstractPreconditioner,
        threshold=1e-4,
    ):
        AbstractAlgoFit.__init__(self, max_iter)

        # step_sizes = create_multi_step_size(list(settings))
        self._step_size = step_size  # [0]

        # heating_list = [ss.heating for ss in step_sizes if ss.heating is not None]

        # self._heating = (
        #     jnp.inf
        #     if len(heating_list) == 0
        #     else max([h for h in heating_list if h is not None])
        # )
        self._heating = (
            self._step_size.heating if self._step_size.heating is not None else jnp.inf
        )

        self._threshold = threshold

        # initial algo parameter
        self._preconditioner = preconditioner

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, step_size):
        self._step_size = step_size

    def get_log_likelihood_kwargs(self, data):
        """return all the needed data"""
        return data

    def results_warper(self, model, data, results, chrono):
        """warp results"""

        out = GDResults.new_from_list(results, chrono)
        # out.reals1d_to_hstack_params(model)
        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        jac_shape = model.jac_log_likelihood(
            theta_reals1d, **log_likelihood_kwargs
        ).shape
        self._preconditioner.initialize(jac_shape)

    # ============================================================== #
    def algorithm(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray = None,
    ):
        """iterative algorithm"""

        for step in itertools.count():

            out = self._algorithm_one_step(
                model, log_likelihood_kwargs, theta_reals1d, step
            )
            theta_reals1d = jnp.where(freezed_components, theta_reals1d, out[0])
            out = (theta_reals1d,) + out[1:]

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta !")
                break

            if jnp.isnan(out[1]).any():
                yield sdg4vsNanError("nan detected in gradient !")
                break

            yield out

            if step > self._heating and jnp.sqrt((out[1] ** 2).sum()) < self._threshold:
                break  # out[1] = grad_precond

    # ============================================================== #
    def _one_gradient_descent(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        # Jacobian
        jac_current = model.jac_log_likelihood(theta_reals1d, **log_likelihood_kwargs)
        # Gradient
        grad = jac_current.mean(axis=0)

        # Preconditionner
        preconditioner, grad_precond = self._preconditioner.get_preconditioned_gradient(
            grad, jac_current, step
        )
        grad_precond *= self._step_size(step)

        theta_reals1d += grad_precond

        return (
            theta_reals1d,
            grad_precond,
            preconditioner,
        )

    def _algorithm_one_step(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        """one iterative algorithm step"""
        # (theta_reals1d, grad_precond, preconditioner)

        return self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )


# if __name__ == "__main__":

#     import matplotlib.pyplot as plt
#     import sdg4varselect.plot as sdgplt
#     from sdg4varselect.models.examples.linear_model import LinearModel
#     from sdg4varselect.outputs import MultiRunRes
#     import jax.random as jrd

#     myModel = LinearModel(50)
#     p_star = myModel.new_params(intercept=1.5, slope=2, sigma2=0.1)
#     p_names = ["intercept", "slope", "sigma2"]

#     obs, sim = myModel.sample(p_star, jrd.PRNGKey(0))
#     plt.plot(obs["time"], obs["Y"], ".")

#     algo_settings = get_gdfim_settings(
#         preheating=2000, heating=6000, learning_rate=1e-3
#     )

#     algoFIM = GradientDescentFIM(
#         2000,
#         algo_settings,
#         preconditioner=Fisher(list(algo_settings)[1:]),
#     )

#     res = []
#     for i in range(1):
#         theta0 = jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
#         res.append(algoFIM.fit(myModel, obs, theta0))
#     res = MultiRunRes(res)
#     print(f"chrono = {res.chrono}")

#     p_emv = myModel.new_params(intercept=1.5, slope=2, sigma2=sim["eps"].var())
#     _ = sdgplt.plot_theta(res, 3, p_emv, p_names)

# algo_settings = [
#     {
#         "learning_rate": 1,
#         "preheating": 0,
#         "heating": 5000,
#         "max": 0.5,
#     }
# ]
# algoAdaGrad = GradientDescentFIM(
#     20000, algo_settings, preconditioner=AdaGrad(regularization=1)
# )
# res = []
# for i in range(10):
#     theta0 = jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
#     res.append(algoAdaGrad.fit(myModel, obs, theta0))
# res = MultiRunRes(res)
# print(f"chrono = {res.chrono}")

# p_emv = myModel.new_params(intercept=1.5, slope=2, sigma2=sim["eps"].var())
# _ = sdgplt.plot_theta(res, 3, p_emv, myModel.params_names)

# _ = sdgplt._plot_theta(
#     jnp.array([res.grad for res in res]).T,
#     params_names=p_names,
#     fig=sdgplt.figure(),
# )

# sdgplt.plt.plot(algoAdaGrad._preconditioner._adagrad_past)
