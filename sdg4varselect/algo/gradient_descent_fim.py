"""
Module for the Gradient Descent algorithm preconditioned by the Fisher Information Matrix.

This module implements a gradient descent algorithm that uses a preconditioner based on
the Fisher Information Matrix to improve convergence.

Created by antoine.caillebotte@inrae.fr
"""

import itertools

import jax.numpy as jnp

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.algo.abstract.abstract_algo_fit import AbstractAlgoFit
from sdg4varselect.algo.preconditioner import AbstractPreconditioner

from sdg4varselect.exceptions import Sdg4vsNanError
from sdg4varselect.outputs import GDResults
from sdg4varselect.learning_rate import LearningRate


class GradientDescentFIM(AbstractAlgoFit):
    """Gradient descent algorithm preconditioned by the Fisher Information Matrix."""

    def __init__(
        self,
        max_iter: int,
        preconditioner: AbstractPreconditioner,
        threshold=1e-4,
    ):
        AbstractAlgoFit.__init__(self, max_iter)

        self._heating = 3500

        self.step_size = LearningRate(
            coef_heating=0.65,
            preheating=1000,
            heating=self._heating,
            coef_preheating=float(jnp.log(1e-8)),
        )

        self._threshold = threshold
        self._preconditioner = preconditioner

    @property
    def step_size(self) -> LearningRate:
        """Return the current step size.

        Returns
        -------
        LearningRate
            current step size"""
        return self._step_size

    @step_size.setter
    def step_size(self, step_size: LearningRate) -> None:
        """Set the step size and update the heating parameter.

        Parameters
        ----------
            step_size: LearningRate
                the new step_size value
        """
        self._step_size = step_size

        self._heating = (
            self._step_size.heating if self._step_size.heating is not None else jnp.inf
        )

    def get_log_likelihood_kwargs(self, data) -> dict:
        """Return all the needed data for the log likelihood computation

        Parameters
        ----------
        data : any
            The data required for log likelihood computation.

        Returns
        -------
        dict
            A dictionary containing the necessary parameters for log likelihood computation.
        """
        return data

    def results_warper(self, model, data, results, chrono) -> GDResults:
        """Warp results into GDResults object.

        Parameters
        ----------
        model : type[AbstractModel]
            The model used for the fitting.
        data : any
            The data used for fitting.
        results : list
            The results obtained from the fitting.
        chrono : timedelta
            The time taken for the fitting.

        Returns
        -------
        GDResults
            An instance of GDResults containing the results.
        """

        out = GDResults.new_from_list(results, chrono)
        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """Initialize the algorithm parameters.

        Parameters
        ----------
        model : type[AbstractModel]
            The model used for fitting.
        log_likelihood_kwargs : dict
            The arguments for computing the log likelihood.
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
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
        """Run the iterative gradient descent algorithm.

        Parameters
        ----------
        model : type[AbstractModel]
            the model to be fitted
        log_likelihood_kwargs : dict
            a dict where all additional log_likelihood arguments can be found
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        freezed_components : jnp.ndarray, optional
            boolean array indicating which parameter components should not be updated (default is None).

        Yields
        ------
        tuple
            A tuple containing the current parameters, gradient, preconditioner.
        """

        for step in itertools.count():

            out = self._algorithm_one_step(
                model, log_likelihood_kwargs, theta_reals1d, step
            )
            theta_reals1d = jnp.where(freezed_components, theta_reals1d, out[0])
            out = (theta_reals1d,) + out[1:]

            if jnp.isnan(theta_reals1d).any():
                yield Sdg4vsNanError("nan detected in theta !")
                break

            if jnp.isnan(out[1]).any():
                yield Sdg4vsNanError("nan detected in gradient !")
                break

            yield out

            if step > self._heating and jnp.sqrt((out[1] ** 2).sum()) < self._threshold:
                break

    # ============================================================== #
    def _one_gradient_descent(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        """Perform one step of gradient descent.

        Parameters
        ----------
        model : type[AbstractModel]
            the model to be fitted
        log_likelihood_kwargs : dict
            a dict where all additional log_likelihood arguments can be found
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        step : int
            The current iteration step.

        Returns
        -------
        tuple
            A tuple containing updated parameters, preconditioned gradient, and the preconditioner.
        """
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
        """Perform one step of the algorithm.

        Parameters
        ----------
        model : type[AbstractModel]
            the model to be fitted
        log_likelihood_kwargs : dict
            a dict where all additional log_likelihood arguments can be found
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        step : int
            The current iteration step.

        Returns
        -------
        tuple
            A tuple containing updated parameters, preconditioned gradient, and the preconditioner.
        """
        return self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )