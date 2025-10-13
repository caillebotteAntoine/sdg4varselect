# pylint: disable = duplicate-code, similar-code
"""
Module for the Stochastic Gradient Descent algorithm preconditioned by the Fisher Information Matrix or another.

This module implements a stochasticgradient descent algorithm that uses a preconditioner based on
the Fisher Information Matrix to improve convergence.
"""

import warnings
from copy import deepcopy

import jax.numpy as jnp

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.exceptions import Sdg4vsException
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    log_likelihood_marginal,
)

from sdg4varselect.algo.gradient_descent_precond import (
    GradientDescentPrecond as GD_Precond,
    BASE_PARAMETERS_GD,
)
from sdg4varselect.algo.abstract.abstract_algo_mcmc import AbstractAlgoMCMC
from sdg4varselect.outputs import SGDResults


from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    AbstractLatentVariablesModel,
)

from sdg4varselect.algo.preconditioner import AbstractPreconditioner
from sdg4varselect._doc_tools import inherit_docstring


@inherit_docstring
class StochasticGradientDescentPrecond(AbstractAlgoMCMC, GD_Precond):
    __doc__ = f"""Stochastic Gradient Descent algorithm preconditioned
    by the Fisher Information Matrix.

    This class implements a Stochastic Gradient Descent (SGD) algorithm, which utilizes
    a preconditioner based on the Fisher Information Matrix (FIM) to improve optimization.
    It integrates elements of the Monte Carlo Markov Chain (MCMC) approach for models with
    latent variables.

    Parameters
    ----------
    {BASE_PARAMETERS_GD}

    Attributes
    ----------
    _pre_heating : int
        Number of initial MCMC sampling steps to stabilize latent variables before optimization.

    Methods
    -------
    get_log_likelihood_kwargs(data):
        Returns a dictionary with required data and latent variables for log likelihood computation.

    results_warper(model, data, results, chrono):
        Formats results into a GDResults object, including marginal likelihood calculation.

    _initialize_algo(model, log_likelihood_kwargs, theta_reals1d):
        Initializes the algorithm's parameters, incorporating a pre-heating phase for MCMC steps.

    _algorithm_one_step(model, log_likelihood_kwargs, theta_reals1d, step):
        Executes a single iteration of SGD, combining MCMC sampling and gradient descent.
    """

    def __init__(self, preconditioner: AbstractPreconditioner, **kwargs):
        GD_Precond.__init__(self, preconditioner, **kwargs)
        AbstractAlgoMCMC.__init__(self)
        self._pre_heating = 1000

    @property
    def pre_heating(self) -> int:
        """Get the number of pre-heating steps for MCMC sampling.
        Returns
        -------
        int
            The number of pre-heating steps."""
        return self._pre_heating

    @pre_heating.setter
    def pre_heating(self, value: int):
        """Set the number of pre-heating steps for MCMC sampling.
        Parameters
        ----------
        value : int
            The number of pre-heating steps to set. Must be a non-negative integer.
        Raises
        ------
        ValueError
            If the value is negative.
        """
        if value < 0:
            raise ValueError("pre_heating must be a non-negative integer.")
        self._pre_heating = value

    def get_log_likelihood_kwargs(  # pylint: disable= missing-return-doc
        self, data
    ) -> dict:
        return data | self.latent_data

    def results_warper(  # pylint: disable= missing-return-doc
        self, model, theta0_reals1d, data, results
    ) -> SGDResults:
        out = SGDResults.new_from_list(results, theta0_reals1d)
        out.latent_variables = {}
        for key, var in self.latent_variables.items():
            out.latent_variables[key] = deepcopy(var.data)

        out.log_likelihood = log_likelihood_marginal(
            model, self._prngkey, data, out.last_theta
        )
        out.reals1d_to_hstack_params(model)

        log_likelihood_kwargs = self.get_log_likelihood_kwargs(data)
        theta_estim = out.last_theta_reals1d

        try:
            out.grad_log_likelihood_marginal = self.grad_log_likelihood_marginal(
                model, log_likelihood_kwargs, theta_estim
            )
            cv_check = jnp.abs(out.grad_log_likelihood_marginal) > 1e-1
            if cv_check.sum() != 0:
                warnings.warn(
                    (
                        f"\nThe algorithm may not have converged, {int(cv_check.sum())} components "
                        "of the gradient of the log marginal likelihood are not sufficiently small"
                    )
                )
        except Sdg4vsException as exc:
            out.grad_log_likelihood_marginal = exc
            print(exc)

        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel] = None,
        theta_reals1d: jnp.ndarray = None,
        freezed_components: jnp.ndarray = None,
        log_likelihood_kwargs: dict = None,
    ) -> None:
        assert isinstance(model, AbstractLatentVariablesModel)
        assert model is not None, "model must be specify !"
        assert (
            log_likelihood_kwargs is not None
        ), "log_likelihood_kwargs must be specify !"
        assert theta_reals1d is not None, "theta_reals1d must be specify !"

        AbstractAlgoMCMC._initialize_algo(self, model)
        GD_Precond._initialize_algo(
            self, model, theta_reals1d, freezed_components, log_likelihood_kwargs
        )
        try:
            for _ in range(self._pre_heating):
                self._one_simulation(log_likelihood_kwargs, theta_reals1d)

        except Sdg4vsException as exc:
            exc.args = (exc.args[0] + " during initialization !",) + exc.args[1:]
            raise exc

    # ============================================================== #
    def breaking_rules(  # pylint: disable= missing-return-doc
        self, step, one_step_results
    ) -> bool:
        first_rules = GD_Precond.breaking_rules(self, step, one_step_results)

        new_rule = True  # jnp.abs(one_step_results[4]).max() < self._threshold

        return first_rules and new_rule

    # ============================================================== #
    def _algorithm_one_step(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        # Simulation
        self._one_simulation(log_likelihood_kwargs, theta_reals1d)

        # Gradient descent
        (
            theta_reals1d,
            grad,
            grad_precond,
            preconditioner,
        ) = self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )

        grad_log_likelihood_marginal = None

        return (
            theta_reals1d,
            grad,
            grad_precond,
            preconditioner,
            grad_log_likelihood_marginal,
        )
