"""
Module for the Gradient Descent algorithm preconditioned by the Fisher Information Matrix or another.

This module implements a gradient descent algorithm that uses a preconditioner based on
the Fisher Information Matrix to improve convergence.

Created by antoine.caillebotte@inrae.fr
"""

from datetime import datetime
from copy import deepcopy

import jax.numpy as jnp

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.algo.abstract.abstract_algo_fit import AbstractAlgoFit
from sdg4varselect.algo.preconditioner import AbstractPreconditioner

from sdg4varselect.exceptions import Sdg4vsNanError, Sdg4vsInfError
from sdg4varselect.outputs import GDResults
from sdg4varselect.learning_rate import LearningRate, cst_step_size


class GradientDescentPrecond(AbstractAlgoFit):
    """Gradient descent algorithm preconditioned by the Fisher Information Matrix or another.

    This class implements a gradient descent algorithm that leverages the Fisher Information Matrix (FIM)
    as a preconditioner to enhance the convergence speed of the optimization process.

    Parameters
    ----------
    preconditioner : AbstractPreconditioner
        An instance of a preconditioner that can computes precondtionned gradient
    threshold : float, optional
        A threshold for the gradient magnitude to determine convergence. The default value is 1e-4.
    max_iter : int
        Maximum number of iterations allowed for the algorithm.
    ntry : int
        Number of attempts to retry the algorithm if an error is encountered.
    ntry_max : int
        Maximum number of retry attempts to reset `_ntry` after each algorithm run.
    partial_fit : bool
        Flag to indicate if partial results should be returned if an error occurs.
    save_all : bool
        Flag to control whether intermediate iterations should be retained.
    save_preconditionner : bool
        Flag to control whether preconditioner values are to be retained
        (warning: risk of using a large amount of memory).


    Attributes
    ----------
    step_size : LearningRate
        The learning rate used for updating the parameters during optimization.

    Methods
    -------
    _one_gradient_descent(model, log_likelihood_kwargs, theta_reals1d, step):
        Executes one step of gradient descent, updating the parameters based on the computed gradients.

    _algorithm_one_step(model, log_likelihood_kwargs, theta_reals1d, step):
        Performs one iteration of the algorithm by invoking the gradient descent step.
    """

    def __init__(
        self, preconditioner: AbstractPreconditioner, threshold=1e-4, **kwargs
    ):
        AbstractAlgoFit.__init__(self, **kwargs)

        self.step_size = deepcopy(cst_step_size)

        self._threshold = threshold
        self._preconditioner = preconditioner
        self._save_preconditioner = False

    @property
    def save_preconditioner(self) -> bool:
        """returns the Flag to control whether preconditioner values are to be retained
        (warning: risk of using a large amount of memory).

        Returns
        -------
            boolean Flag to control whether preconditioner values are to be retained
        """
        return self._save_preconditioner

    @save_preconditioner.setter
    def save_preconditioner(self, save_preconditioner: bool):
        self._save_preconditioner = save_preconditioner

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
        self._step_size = step_size

        self._heating = (
            self._step_size.heating.step
            if self._step_size.heating.step is not None
            else jnp.inf
        )

    # ============================================================== #
    def results_warper(self, model, data, results, chrono) -> GDResults:
        """Warp results into Sdg4vsResults object.

        Parameters
        ----------
        model : type[AbstractModel]
            The model used for the fitting.
        data : dict
           a dict where all additional log_likelihood arguments can be found
        results : list
            The results obtained from the fitting.
        chrono : timedelta
            The time taken for the fitting.

        Returns
        -------
        Sdg4vsResults
            An instance of Sdg4vsResults containing the results.
        """
        chrono_start = datetime.now()
        out = GDResults.new_from_list(results, chrono)
        out.log_likelihood = model.log_likelihood(out.last_theta, data)
        out.reals1d_to_hstack_params(model)
        out.fim = self._preconditioner.preconditioner
        out.chrono += datetime.now() - chrono_start
        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        theta_reals1d: jnp.ndarray,
        log_likelihood_kwargs: dict,
    ) -> None:
        jac_shape = model.jac_log_likelihood(
            theta_reals1d, **log_likelihood_kwargs
        ).shape
        self._preconditioner.initialize(jac_shape)

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
            A tuple containing updated parameters, gradient, and the preconditioner.
        """
        # Jacobian
        jac_current = model.jac_log_likelihood(theta_reals1d, **log_likelihood_kwargs)
        # Gradient
        grad = jac_current.mean(axis=0)

        # Preconditionner
        grad_precond = self._preconditioner.get_preconditioned_gradient(
            grad, jac_current, step
        )
        grad_precond *= self._step_size(step)

        theta_reals1d += grad_precond

        if jnp.isnan(theta_reals1d).any():
            raise Sdg4vsNanError("nan detected in theta !")

        if jnp.isnan(grad_precond).any():
            raise Sdg4vsNanError("nan detected in gradient !")

        if jnp.isinf(theta_reals1d).any():
            raise Sdg4vsInfError("inf detected in theta !")

        if jnp.isinf(grad_precond).any():
            raise Sdg4vsInfError("inf detected in gradient !")

        preconditioner = (
            self._preconditioner.value if self._save_preconditioner else None
        )
        return (theta_reals1d, grad, grad_precond, preconditioner)

    # ============================================================== #
    def breacking_rules(self, step, one_step_results):
        """Determine whether to stop the optimization process.

        This function checks if the stopping criteria are met based on the number of iterations
        and the norm of the gradient.

        Parameters
        ----------
        step : int
            The current iteration step.
        one_step_results : tuple
            The tuple returned by the _algorithm_one_step function

        Returns
        -------
        bool
            True if the stopping conditions are met, otherwise False.
        """
        return (
            step > self._heating
            and jnp.sqrt((one_step_results[1] ** 2).sum()) < self._threshold
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
        (theta_reals1d, grad, preconditioner) = self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )
        return (theta_reals1d, grad, preconditioner)
