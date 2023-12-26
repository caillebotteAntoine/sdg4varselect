import jax.numpy as jnp


import parametrization_cookbook.jax as pc

from sdg4varselect.MCMC import MCMC_chain

# import numpy as np

# class theta_handler:
#     def __init__(self, params, parametrization):
#         """Constructor of ."""
#         self._reals1d: jnp.ndarray = None
#         self._parametrization: pc.NamedTuple = parametrization

#         self.reals1d = params

#     # @property
#     # def parametrization(self) -> pc._common.NamedTuple:
#     #     """returns theta parametrization"""
#     #     return self._parametrization

#     @property
#     def params(self):
#         """returns the last estimated value of the parameters"""
#         return self._parametrization.reals1d_to_params(self._reals1d)

#     # def parametrization_reals1d_to_params(self, x):
#     #     return self._parametrization.reals1d_to_params(x)

#     @property
#     def reals1d(self):
#         """returns theta in the reparameterized space"""
#         return self._reals1d

#     @reals1d.setter
#     def reals1d(self, kwargs):
#         if self._parametrization is None:
#             raise ValueError("parametrization must be initiate first.")

#         if isinstance(kwargs, jnp.ndarray):
#             self._reals1d = kwargs
#         elif isinstance(kwargs, dict):
#             self._reals1d = self._parametrization.params_to_reals1d(**kwargs)
#         else:
#             self._reals1d = self._parametrization.params_to_reals1d(kwargs)

#     @property
#     def params_names(self):
#         idx_params = self._parametrization.idx_params
#         rep_num = [idx.stop - idx.start for idx in idx_params]
#         names = idx_params._fields
#         return np.repeat(names, rep_num)

#     def theta_nonzero_support(self, p, theta=None):
#         """return the mask of non zero last p component of theta"""
#         if theta is None:
#             theta = self._reals1d

#         d = len(theta) - p
#         LD_mask = [True for k in range(d)]
#         HD_mask = jnp.arange(len(theta)) >= d
#         return jnp.hstack([LD_mask, theta[HD_mask] != 0])

#     def get_number_of_nonzero(self, p):
#         d = len(self._reals1d) - p
#         return self.theta_nonzero_support(p=p).sum() - d

#     def __repr__(self) -> str:
#         msg = "[ ==  === ]\n"

#         msg += (
#             str(self.params)
#             .replace("Array", "")
#             .replace(", dtype=float32", "")
#             .replace(", dtype=float64", "")
#             .replace(", weak_type=True", "")
#             .replace("Parameters(", "\n\t*parameters :\n\t\t")
#             .replace(", ", ",\n\t\t")
#         )

#         return msg


class Data_handler:
    def __init__(self):
        """Constructor of ."""
        self.latent_variables: dict[str, MCMC_chain] = {}
        self._data = {}

    @property
    def data(self):
        return self._data

    def add_data(self, **kwargs) -> None:
        """adds variables to the solver data"""
        for key, item in kwargs.items():
            if key in self._data:
                raise KeyError(key + " all ready exist in solver's data.")
            self._data[key] = item

    def update_data(self, **kwargs) -> None:
        """update variables to the solver data"""
        for key, item in kwargs.items():
            if key in self.latent_variables:
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
        """create a new mcmc chain and add it to the latent variable of the solver"""
        new_mcmc = MCMC_chain(*args, **kwargs)
        new_mcmc_name = new_mcmc.name
        if new_mcmc_name in self.latent_variables:
            raise KeyError(
                new_mcmc_name + " all ready exist in solver's latent_variables."
            )
        self.latent_variables[new_mcmc_name] = new_mcmc
        self.add_data(**dict(((new_mcmc_name, new_mcmc.data),)))
