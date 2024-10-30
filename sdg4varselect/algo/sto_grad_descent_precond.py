"""
Module for the Stochastic Gradient Descent algorithm preconditioned by the Fisher Information Matrix or another.

This module implements a stochasticgradient descent algorithm that uses a preconditioner based on
the Fisher Information Matrix to improve convergence.

Created by antoine.caillebotte@inrae.fr
"""

from datetime import datetime

import jax.numpy as jnp

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    log_likelihood_marginal,
)

from sdg4varselect.algo.gradient_descent_precond import (
    GradientDescentPrecond as GD_Precond,
)
from sdg4varselect.algo.abstract.abstract_algo_mcmc import AbstractAlgoMCMC
from sdg4varselect.outputs import GDResults


from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    AbstractLatentVariablesModel,
)

from sdg4varselect.algo.preconditioner import AbstractPreconditioner


class StochasticGradientDescentPrecond(AbstractAlgoMCMC, GD_Precond):
    """Stochastic Gradient Descent algorithm preconditioned
    by the Fisher Information Matrix.

    This class implements a Stochastic Gradient Descent (SGD) algorithm, which utilizes
    a preconditioner based on the Fisher Information Matrix (FIM) to improve optimization.
    It integrates elements of the Monte Carlo Markov Chain (MCMC) approach for models with
    latent variables..

    Parameters
    ----------
    prngkey : jax.random.PRNGKey
        A PRNG key, consumable by random functions.
    max_iter : int
        Maximum number of iterations allowed for the algorithm.

    preconditioner : AbstractPreconditioner
        An instance of a preconditioner that can computes precondtionned gradient

    threshold : float, optional
        A threshold for the gradient magnitude to determine convergence. The default value is 1e-4.

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

    def __init__(
        self,
        prngkey,
        max_iter: int,
        preconditioner: AbstractPreconditioner,
        threshold=1e-4,
    ):
        GD_Precond.__init__(self, max_iter, preconditioner, threshold)
        AbstractAlgoMCMC.__init__(self, prngkey)
        self._pre_heating = 1000

    def get_log_likelihood_kwargs(self, data):
        """Return all the needed data, like latent variables for the log likelihood computation

        Parameters
        ----------
        data : any
            The data required for log likelihood computation.

        Returns
        -------
        dict
            A dictionary containing the necessary data, like latent variables for log likelihood computation.
        """
        return data | self.latent_data

    def results_warper(self, model, data, results, chrono):
        """Warp results into Sdg4vsResults object and calculate marginal likelihood.

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
            An instance of Sdg4vsResults containing the results, including marginal likelihood.
        """
        chrono_start = datetime.now()

        out = GDResults.new_from_list(results, chrono)

        out.likelihood = log_likelihood_marginal(
            model, self._prngkey, data, out.last_theta
        )
        out.reals1d_to_hstack_params(model)
        out.chrono += datetime.now() - chrono_start
        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel] = None,
        log_likelihood_kwargs: dict = None,
        theta_reals1d: jnp.ndarray = None,
    ) -> None:
        assert isinstance(model, AbstractLatentVariablesModel)
        assert model is not None, "model must be specify !"
        assert (
            log_likelihood_kwargs is not None
        ), "log_likelihood_kwargs must be specify !"
        assert theta_reals1d is not None, "theta_reals1d must be specify !"

        AbstractAlgoMCMC._initialize_algo(self)
        GD_Precond._initialize_algo(self, model, log_likelihood_kwargs, theta_reals1d)

        for _ in range(self._pre_heating):
            self._one_simulation(log_likelihood_kwargs, theta_reals1d)

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
        return self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )