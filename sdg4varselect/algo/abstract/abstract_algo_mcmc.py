"""
Module for Abstract class containing common methods for algorithms based on Monte Carlo Markov Chains (MCMC).

This module provides an abstract base class `AbstractAlgoMCMC` for MCMC-based algorithms, with utilities for
initializing MCMC chains, handling latent variables, and performing MCMC sampling.

Created by antoine.caillebotte@inrae.fr
"""

from functools import partial

from jax import jit
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

    def _initialize_algo(self) -> None:
        """Initialize MCMC-related components of the algorithm."""
        for var in self.latent_variables.values():
            var.reset()

    # ============================================================== #
    def init_mcmc(
        self,
        theta0,
        model: type[AbstractLatentVariablesModel],
        sd: dict[str, float] = None,
    ):
        """Initialize MCMC chains based on a model's latent variables.

        Parameters
        ----------
        theta0 : jnp.ndarray
            Initial values for model parameters.
        model : type[AbstractLatentVariablesModel]
            The model with latent variable definitions.
        sd : dict[str, float], optional
            Standard deviation values for each MCMC chain (default is None).

        Raises
        ------
        KeyError
            If a latent variable with the same name already exists.
        """

        params0 = model.parametrization.reals1d_to_params(theta0)
        for new_mcmc_name in model.latent_variables_name:
            data = model.latent_variables_data(params0, new_mcmc_name)

            self.add_mcmc(
                data["mean"],
                sd=1 if sd is None else sd[new_mcmc_name],
                size=data["size"],
                likelihood=model.log_likelihood_array,
                name=new_mcmc_name,
            )

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

    @partial(jit, static_argnums=0)
    def _one_simulation(self, likelihood_kwargs, theta_reals1d):
        """Perform one simulation step for each latent variable.

        Parameters
        ----------
        likelihood_kwargs : dict
            Additional keyword arguments needed for likelihood calculations.
        theta_reals1d : jnp.ndarray
            Current parameter values for the model.
        """
        for var_lat in self._latent_variables.values():
            self._prngkey = var_lat.gibbs_sampler_step(
                self._prngkey, theta_reals1d, **likelihood_kwargs
            )

    # ============================================================== #
