"""
Module for Abstract class containing common methods for algorithms based on Monte Carlo Markov Chains (MCMC).

This module provides an abstract base class `AbstractAlgoMCMC` for MCMC-based algorithms, with utilities for
initializing MCMC chains, handling latent variables, and performing MCMC sampling.

Created by antoine.caillebotte@inrae.fr
"""

from typing import Type

import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect._mcmc import MCMC
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    AbstractLatentVariablesModel,
)


class AbstractAlgoMCMC:
    """Abstract base class for implementing algorithms based on Monte Carlo Markov Chains (MCMC).

    This class provides methods for initializing and managing MCMC chains, handling latent variables,
    and updating simulation data for MCMC-based algorithms. It facilitates the setup and execution
    of MCMC sampling by creating and storing latent variables as MCMC instances, allowing algorithm
    extensions for models that use MCMC in their fitting procedures.

    Attributes
    ----------
    _prngkey : jax.random.PRNGKey
        A PRNG key, consumable by random functions.
    _latent_variables : dict[str, MCMC]
        Dictionary of latent variable names mapped to their corresponding MCMC instances.
    _latent_data : dict
        Dictionary containing additional data associated with each latent variable for MCMC sampling.

    Methods
    -------
    set_seed(prngkey)
        Sets the random key to ensure consistent randomness in MCMC processes.
    _initialize_algo(model, likelihood_kwargs, theta_reals1d)
        Initializes algorithm-related settings for latent variables based on model specifications.
    init_mcmc(theta0, model, sd=None)
        Initializes MCMC chains for latent variables as defined in the model.
    add_data(**kwargs)
        Adds custom data variables for MCMC sampling.
    add_mcmc(*args, **kwargs)
        Creates and adds a new MCMC chain to the latent variable dictionary.
    _one_simulation(likelihood_kwargs, theta_reals1d)
        Executes one simulation step for each latent variable in the MCMC chains.
    """

    def __init__(self):
        self._prngkey = jrd.PRNGKey(0)
        self._latent_variables: dict[str, MCMC] = {}
        self._latent_data: dict[str, jnp.ndarray] = {}
        self.n_simulation = 1

    @property
    def latent_variables(self) -> dict[str, MCMC]:
        """Return the dictionary of latent MCMC variables.

        Returns
        -------
        dict[str, MCMC]
            A dictionary where keys are variable names and values are MCMC instances.
        """
        return self._latent_variables

    @property
    def latent_data(self) -> dict[str, jnp.ndarray]:
        """Return the dictionary of latent data used in MCMC.

        Returns
        -------
        dict
            A dictionary containing data associated with latent variables.
        """
        return self._latent_data

    def set_seed(self, prngkey) -> None:
        """Update the random key for reproducibility in stochastic processes.

        Parameters
        ----------
        prngkey : jax.random.PRNGKey
            A PRNG key, consumable by random functions.
        """
        self._prngkey = prngkey

    def _initialize_algo(
        self,
        model: type[AbstractLatentVariablesModel],
    ) -> None:
        """Initialize the algorithm by reseting MCMC chains to a specific starting value.

        Parameters
        ----------
        model : type[AbstractLatentVariablesModel]
            The model with latent variable definitions.
        """
        for _, var in self.latent_variables.items():
            var.reset(x0=0)
            var.likelihood = model.log_likelihood_array

    # ============================================================== #
    def init_mcmc(
        self,
        model: type[AbstractLatentVariablesModel],
        sd: dict[str, float] = None,
        adaptative_sd=True,
    ):
        """Initialize MCMC chains based on a model's latent variables.

        Parameters
        ----------
        model : type[AbstractLatentVariablesModel]
            The model with latent variable definitions.
        sd : dict[str, float], optional
            Standard deviation values for each MCMC chain (default is None).
        adaptative_sd : bool, optional
            A flag indicating whether to use adaptive standard deviation adjustments


        Raises
        ------
        KeyError
            If a latent variable with the same name already exists.

        TODO check if mcmc have been init
        """
        for new_mcmc_name in model.latent_variables_name:
            self.add_mcmc(
                likelihood=model.log_likelihood_array,
                sd=1 if sd is None else sd[new_mcmc_name],
                x0=0,
                size=model.latent_variables_size,
                name=new_mcmc_name,
            )

        for var_lat in self.latent_variables.values():
            var_lat.adaptative_sd = adaptative_sd

    def add_data(self, **kwargs) -> None:
        """Add variables to the MCMC solver data.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs where keys are variable names and values are data items.

        Raises
        ------
        KeyError
            If a variable with the same key already exists in the solver's data.
        """
        for key, item in kwargs.items():
            if key in self._latent_data:
                raise KeyError(key + " all ready exist in solver's data.")
            self._latent_data[key] = item

    def add_mcmc(self, *args, **kwargs) -> None:
        """Create a new MCMC chain and add it to the latent variables dictionary.

        Parameters
        ----------
        *args : tuple
            Positional arguments for creating an MCMC instance.
        **kwargs : dict
            Keyword arguments for creating an MCMC instance.

        Raises
        ------
        KeyError
            If an MCMC chain with the same name already exists.
        """
        new_mcmc = MCMC(*args, **kwargs)
        new_mcmc_name = new_mcmc.name
        if new_mcmc_name in self._latent_variables:
            raise KeyError(
                new_mcmc_name + " all ready exist in solver's latent_variables."
            )
        self._latent_variables[new_mcmc_name] = new_mcmc
        self.add_data(**dict(((new_mcmc_name, new_mcmc.data),)))

    # ============================================================== #
    def _one_simulation(self, likelihood_kwargs, theta_reals1d):
        """Perform one or several simulation step for each latent variable.

        Parameters
        ----------
        likelihood_kwargs : dict
            Additional keyword arguments needed for likelihood calculations.
        theta_reals1d : jnp.ndarray
            Current parameter values for the model.
        """
        for _ in range(self.n_simulation):
            for var_lat in self._latent_variables.values():
                self._prngkey = var_lat.gibbs_sampler_step(
                    self._prngkey, theta_reals1d=theta_reals1d, **likelihood_kwargs
                )

    # ============================================================== #

    def grad_log_likelihood_marginal(
        self,
        model: Type[AbstractLatentVariablesModel],
        log_likelihood_kwargs: dict,
        theta_reals1d: jnp.ndarray,
        size=300,
    ) -> jnp.ndarray:
        """
        Compute the marginal log-likelihood.

        Parameters
        ----------
        model : Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]]
            Model instance.
        log_likelihood_kwargs : dict
            a dict where all log_likelihood arguments can be found.
        theta_reals1d : jnp.ndarray
            Parameters passed to the log-likelihood function.
        size : int, optional
            Number of simulations for marginalization, by default 1000.

        Returns
        -------
        jnp.ndarray
            Marginal log-likelihood.
        """

        out = []
        for _ in range(size):
            # Simulation
            self._one_simulation(log_likelihood_kwargs, theta_reals1d)
            # Jacobian
            jac = model.jac_log_likelihood(theta_reals1d, **log_likelihood_kwargs)
            # Gradient
            out.append(jac.mean(axis=0))

        grad = jnp.array(out).mean(axis=0)
        grad_old = jnp.array(out[:-2]).mean(axis=0)

        n_simu = len(out)
        while n_simu < size * 2 and (abs(grad / grad_old - 1.0) >= 1e-2).all():
            print(abs(grad - grad_old))
            for _ in range(100):
                n_simu += 1
                # Simulation
                self._one_simulation(log_likelihood_kwargs, theta_reals1d)
                # Jacobian
                jac = model.jac_log_likelihood(theta_reals1d, **log_likelihood_kwargs)
                # Gradient
                out.append(jac.mean(axis=0))

                grad_old = grad
                grad = jnp.array(out).mean(axis=0)

        return grad
