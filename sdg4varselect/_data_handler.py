"""
Module for DataHandler class.

Create by antoine.caillebotte@inrae.fr
"""
from copy import deepcopy
from sdg4varselect._MCMC import MCMC_chain


class DataHandler:
    """
    A class for handling data and latent variables in a algorithm.

    Attributes:
    ----------
        _latent_variables (dict):
            A dictionary storing MCMC chains as latent variables.
        _data (dict):
            A dictionary storing additional solver data.

    Methods:
    ----------
        __init__(self):
            Constructor method to initialize an instance of DataHandler.

        deepcopy(self):
            Creates a deep copy of the current DataHandler instance.

        add_data(self,**kwargs):
            Adds variables to the solver data.

        update_data(self,**kwargs):
            Updates variables in the solver data, excluding latent variables.

        add_mcmc(self,*args, **kwargs):
            Creates a new MCMC chain and adds it to the latent variables.

    Properties:
    ----------
        latent_variables->dict[str, MCMC_chain]:
            Returns the latent variables dictionary.

        data->dict:
            Returns the data dictionary.
    """

    def __init__(self, **kwargs):
        """
        Constructor method for the DataHandler class.
        Initializes _latent_variables and _data dictionaries.
        """
        self._latent_variables: dict[str, MCMC_chain] = {}
        self._data = {}

        self.add_data(**kwargs)

    @property
    def latent_variables(self) -> dict[str, MCMC_chain]:
        """Returns the latent variables dictionary."""
        return self._latent_variables

    @property
    def data(self) -> dict:
        """Returns the data dictionary."""
        return self._data

    # ============================================================== #
    def deepcopy(self):
        """Creates a deep copy of the current DataHandler instance."""
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
