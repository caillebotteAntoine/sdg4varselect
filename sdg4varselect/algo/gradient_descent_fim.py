"""
Module for Gradient descent algorithm preconditioned by the fisher information matrix.

Create by antoine.caillebotte@inrae.fr
"""

import itertools
from collections import namedtuple

import jax.numpy as jnp

from sdg4varselect.learning_rate import create_multi_step_size
from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.algo.abstract.abstract_algo_fit import AbstractAlgoFit
from sdg4varselect.outputs import GDResults

from sdg4varselect.algo.stochastic_gradient_descent_utils import (
    gradient_descent_fisher_preconditionner,
)


GradientDescentFIMSettings = namedtuple(
    "GradientDescentFIMSettings",
    ("step_size_grad", "step_size_approx_sto", "step_size_fisher"),
)


def get_GDFIM_settings(preheating, heating, learning_rate=1e-8):
    return GradientDescentFIMSettings(
        step_size_grad={
            "learning_rate": learning_rate,
            "preheating": preheating,
            "heating": heating,
            "max": 0.9,
        },
        step_size_approx_sto={
            "learning_rate": learning_rate,
            "preheating": preheating,
            "heating": None,
            "max": 1,
        },
        step_size_fisher={
            "learning_rate": learning_rate,
            "preheating": preheating,
            "heating": None,
            "max": 0.9,
        },
    )


class GradientDescentFIM(AbstractAlgoFit):
    """Gradient descent algorithm preconditioned by the fisher information matrix"""

    def __init__(
        self,
        max_iter: int,
        settings: GradientDescentFIMSettings,
    ):
        AbstractAlgoFit.__init__(self, max_iter)

        step_sizes = create_multi_step_size(list(settings), num_step_size=3)
        (
            self._step_size_grad,
            self._step_size_approx_sto,
            self._step_size_fisher,
        ) = (
            step_sizes[0],
            step_sizes[1],
            step_sizes[2],
        )

        heating_list = [
            settings.step_size_approx_sto["heating"],
            settings.step_size_fisher["heating"],
            settings.step_size_grad["heating"],
        ]

        self._heating = (
            jnp.inf
            if len(heating_list) == 0
            else max([h for h in heating_list if h is not None])
        )

        # initial algo parameter
        self._jac = jnp.zeros(shape=(1, 1))

    def get_likelihood_kwargs(self, data):
        """return all the needed data"""
        return data

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        jac_shape = model.jac_likelihood(theta_reals1d, **likelihood_kwargs).shape
        self._jac = jnp.zeros(shape=jac_shape)

    # ============================================================== #
    def _one_gradient_descent(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        jac: jnp.ndarray,
        step: int,
    ):
        step_size = [
            self._step_size_grad(step),
            self._step_size_approx_sto(step),
            self._step_size_fisher(step),
        ]

        # Gradient descent
        jac_current = model.jac_likelihood(theta_reals1d, **likelihood_kwargs)

        (jac, fisher_info, grad_precond) = gradient_descent_fisher_preconditionner(
            self._jac,
            jac_current,
            step_size_approx_sto=step_size[1],
            step_size_fisher=step_size[2],
        )

        grad_precond *= step_size[0]
        theta_reals1d += grad_precond

        return (
            theta_reals1d,
            jac,
            fisher_info,
            grad_precond,
        )

    def algorithm(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ):
        """iterative algorithm"""

        for step in itertools.count():

            (theta_reals1d, self._jac, fisher_info, grad_precond) = (
                self._one_gradient_descent(
                    model, likelihood_kwargs, theta_reals1d, self._jac, step
                )
            )

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta or jac")
                break

            yield (theta_reals1d, fisher_info, grad_precond)

            if step > self._heating and jnp.sqrt((grad_precond**2).sum()) < 1e-3:
                break

    def results_warper(self, model, data, results, chrono):
        """warp results"""

        out = GDResults.new_from_list(results, chrono)
        return GDResults.compute_with_model(model, out)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sdg4varselect.models.linear_model import LinearModel
    from sdg4varselect.outputs import MultiRunRes
    import jax.random as jrd

    myModel = LinearModel(150)
    p_star = myModel.new_params(intercept=1.5, slope=0.5, sigma2=0.1)

    obs, sim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.plot(obs["time"], obs["Y"], ".")

    algo_settings = get_GDFIM_settings(preheating=400, heating=600)

    algo = GradientDescentFIM(1000, algo_settings)

    res = []
    for i in range(10):
        theta0 = jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(algo.fit(myModel, obs, theta0))
    res = MultiRunRes(res)

    import sdg4varselect.plot as sdgplt

    sdgplt.plot_theta(res, 3, p_star, myModel.params_names)
    print(f"chrono = {res.chrono}")
