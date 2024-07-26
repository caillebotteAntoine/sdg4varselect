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

from sdg4varselect.algo.preconditioner import (
    Fisher,
    AdaGrad,
    AbstractPreconditioner,
)


GradientDescentFIMSettings = namedtuple(
    "GradientDescentFIMSettings",
    ("step_size_grad", "step_size_approx_sto", "step_size_fisher"),
)


def get_gdfim_settings(preheating, heating, learning_rate=1e-8):
    """return 3 step size well parametred"""
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
        preconditioner: AbstractPreconditioner,
    ):
        AbstractAlgoFit.__init__(self, max_iter)

        step_sizes = create_multi_step_size(list(settings))
        self._step_size_grad = step_sizes[0]

        heating_list = [ss.heating for ss in step_sizes if ss.heating is not None]

        self._heating = (
            jnp.inf
            if len(heating_list) == 0
            else max([h for h in heating_list if h is not None])
        )

        self._threshold = 1e-4

        # initial algo parameter
        self._preconditioner = preconditioner

    def get_log_likelihood_kwargs(self, data):
        """return all the needed data"""
        return data

    def results_warper(self, model, data, results, chrono):
        """warp results"""

        out = GDResults.new_from_list(results, chrono)
        out.reals1d_to_hstack_params(model)
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
    ):
        """iterative algorithm"""

        for step in itertools.count():

            out = self._algorithm_one_step(
                model, log_likelihood_kwargs, theta_reals1d, step
            )
            theta_reals1d = out[0]

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta or jac")
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

        grad_precond *= self._step_size_grad(step)

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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.linear_model import LinearModel
    from sdg4varselect.outputs import MultiRunRes
    import jax.random as jrd

    myModel = LinearModel(50)
    p_star = myModel.new_params(intercept=1.5, slope=2, sigma2=0.1)

    obs, sim = myModel.sample(p_star, jrd.PRNGKey(0))
    # obs["time"] = jnp.array(
    #     [5, 10, 12, 14, 16, 25, 30, 50, 150, 200, 250, 300], dtype=jnp.float64
    # )
    # myModel = LinearModel(obs["time"].shape[0])
    # obs["Y"] = (p_star.slope) * obs["time"]
    # obs["Y"] = jnp.array(
    #     [
    #         0.28,
    #         0.33,
    #         0.33,
    #         0.35,
    #         0.47,
    #         0.43,
    #         0.51,
    #         0.65,
    #         3.44,
    #         4.28,
    #         6.14,
    #         7.04,
    #     ]
    # )

    # obs["Y"] = obs["Y"].at[5].set(0.35)

    plt.plot(obs["time"], obs["Y"], ".")

    algo_settings = get_gdfim_settings(
        preheating=2000, heating=6000, learning_rate=1e-3
    )

    algoFIM = GradientDescentFIM(
        2000,
        algo_settings,
        preconditioner=Fisher(list(algo_settings)[1:]),
    )

    algo_settings = [
        {
            "learning_rate": 1,
            "preheating": 0,
            "heating": 5000,
            "max": 0.5,
        }
    ]
    algoAdaGrad = GradientDescentFIM(
        20000, algo_settings, preconditioner=AdaGrad(regularization=1)
    )

    res = []
    for i in range(10):
        theta0 = jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(algoFIM.fit(myModel, obs, theta0))
    res = MultiRunRes(res)
    print(f"chrono = {res.chrono}")

    p_emv = myModel.new_params(intercept=1.5, slope=2, sigma2=sim["eps"].var())
    _ = sdgplt.plot_theta(res, 3, p_emv, myModel.params_names)

    res = []
    for i in range(10):
        theta0 = jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(algoAdaGrad.fit(myModel, obs, theta0))
    res = MultiRunRes(res)
    print(f"chrono = {res.chrono}")

    p_emv = myModel.new_params(intercept=1.5, slope=2, sigma2=sim["eps"].var())
    _ = sdgplt.plot_theta(res, 3, p_emv, myModel.params_names)

    _ = sdgplt._plot_theta(
        jnp.array([res.grad for res in res]).T,
        params_names=myModel.params_names,
        fig=sdgplt.figure(),
    )

    sdgplt.plt.plot(algoAdaGrad._preconditioner._adagrad_past)
