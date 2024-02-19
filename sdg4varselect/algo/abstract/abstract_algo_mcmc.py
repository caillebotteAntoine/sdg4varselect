"""
Module for Abstract class containing common method for algorithm based on Monte-Carlo Markov Chains.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0613
from copy import deepcopy
import jax.numpy as jnp

from sdg4varselect._MCMC import MCMC_chain
from sdg4varselect.models.abstract.abstract_model import AbstractModel


class AbstractAlgoMCMC:
    """Abstract class containing common method for algorithm based on Monte-Carlo Markov Chains."""

    def __init__(self, prngkey):
        self._prngkey = prngkey
        self._latent_variables: dict[str, MCMC_chain] = {}
        self._latent_data = {}

    @property
    def latent_variables(self) -> dict[str, MCMC_chain]:
        """Returns the latent variables dictionary."""
        return self._latent_variables

    @property
    def latent_data(self) -> dict:
        """Returns the latent_data dictionary."""
        return self._latent_data

    def set_seed(self, prngkey) -> None:
        self._prngkey = prngkey

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        for var in self.latent_variables.values():
            var.reset()

    # ============================================================== #
    def add_data(self, **kwargs) -> None:
        """adds variables to the solver data"""
        for key, item in kwargs.items():
            if key in self._latent_data:
                raise KeyError(key + " all ready exist in solver's data.")
            self._latent_data[key] = item

    def add_mcmc(self, *args, **kwargs) -> None:
        """create a new mcmc chain and add it to the latent variable"""
        new_mcmc = MCMC_chain(*args, **kwargs)
        new_mcmc_name = new_mcmc.name
        if new_mcmc_name in self._latent_variables:
            raise KeyError(
                new_mcmc_name + " all ready exist in solver's latent_variables."
            )
        self._latent_variables[new_mcmc_name] = new_mcmc
        self.add_data(**dict(((new_mcmc_name, new_mcmc.data),)))

    # ============================================================== #
    def likelihood_marginal(self, model, data, theta, size=1000):
        var_lat_sample = {}
        for key, item in self.latent_variables.items():
            var_lat_sample[key] = item.sample(
                self._prngkey,
                theta,
                size=size,
                **data,
                **self.latent_data,
            )

        likelihood_kwargs = deepcopy(self.latent_data)
        out = [model.likelihood(theta, **data, **likelihood_kwargs)]

        def add_val(k):
            for var in self.latent_variables:
                likelihood_kwargs[var] = var_lat_sample[var][k]

            new_val = model.likelihood(theta, **data, **likelihood_kwargs)
            out.append(out[-1] + new_val)

        for k in range(1, 10):
            add_val(k)

        n_simu = 10
        while n_simu < size and abs(out[-2] / (n_simu - 1) - out[-1] / n_simu) >= 1e-3:
            add_val(n_simu)
            n_simu += 1
        return out[-1] / n_simu

    # ============================================================== #

    # @functools.partial(jit, static_argnums=0)
    def _one_simulation(self, likelihood_kwargs, theta_reals1d):
        # Simulation
        for var_lat in self._latent_variables.values():
            self._prngkey = var_lat.gibbs_sampler_step(
                self._prngkey, theta_reals1d, **likelihood_kwargs
            )

    # ============================================================== #


if __name__ == "__main__":
    pass
