import jax.numpy as jnp
import parametrization_cookbook.jax as pc
from copy import deepcopy

from sdg4varselect.MCMC import MCMC_chain


class Data_handler:
    def __init__(self):
        """Constructor of ."""
        self._latent_variables: dict[str, MCMC_chain] = {}
        self._data = {}

    @property
    def latent_variables(self):
        return self._latent_variables

    @property
    def data(self):
        return self._data

    # ============================================================== #
    def deepcopy(self):
        return deepcopy(self)

    def add_data(self, **kwargs) -> None:
        """adds variables to the solver data"""
        for key, item in kwargs.items():
            if key in self._data:
                raise KeyError(key + " all ready exist in solver's data.")
            self._data[key] = item

    def update_data(self, **kwargs) -> None:
        """update variables to the solver data"""
        for key, item in kwargs.items():
            if key in self._latent_variables:
                raise KeyError(
                    f"changing the value of a latent variable ({key}) is not allowed."
                )

            if key in self._likelihood_kwargs:
                self._likelihood_kwargs[key] = item

            if key in self._data:
                self._data[key] = item
            else:
                raise KeyError(f"{key} does not exist in global variables.")

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
