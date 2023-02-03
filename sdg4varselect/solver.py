from typing import Optional
from warnings import warn

import numpy as np
import parametrization_cookbook.jax as pc

# from chain import chain
from sdg4varselect.MCMC import MCMC_chain
from sdg4varselect.parameter import parameter


class solver:
    def __init__(self):
        """Constructor of solver."""
        self.parameters = {}
        self._theta = {}
        self._theta_parametrization: pc.NamedTuple = None
        self.latent_variables = {}
        self.__global_variables = {}

        self._data = {}

        self.__is_init = False

    def is_init(self):
        return self.__is_init

    def theta_to_params(self):
        theta_array = np.concatenate([x for x in self._theta])
        theta = self._theta_parametrization.reals1d_to_params(theta_array)
        return theta

    def theta(self):
        return self._theta

    def data(self):
        return self._data

    def get_global(self, name):
        if name in self.__global_variables:
            return self.__global_variables[name]
        raise KeyError(name + " isn't in global variables")

    def set_data(self, *name_args) -> None:
        if len(name_args) > 1:
            for n in name_args:
                self.set_data(n)
        else:
            name = name_args[0]
            if isinstance(name, str):
                if name in self.parameters:
                    self._data[name] = self.parameters[name].data()
                elif name in self.latent_variables:
                    self._data[name] = self.latent_variables[name].data()
                elif name in self.__global_variables:
                    self._data[name] = self.__global_variables[name]
                else:
                    raise KeyError(
                        name
                        + "does not exist neiter in parameters or in latent variables or in global variables"
                    )
            else:
                raise TypeError("name must be a str or a list of str")

    def add_variable(self, name: str, x) -> None:
        self.__global_variables[name] = x

    def add_parameter(self, x: parameter) -> None:
        name = x.name()
        if name == "NA":
            raise ValueError("parameters must have a name to be added to the solver")

        self.parameters[name] = x
        self._theta[name] = x.data()

    def init_parameters(self):
        for par in self.parameters.values():

            name = par.linked_name
            if name in self.latent_variables:
                var = self.latent_variables[name].data()
            elif name in self.__global_variables:
                var = self.__global_variables[name]
            else:
                raise KeyError(
                    name
                    + " does not exist neiter in latent variables or in global variables"
                )

            par.init(var)

        from sdg4varselect.miscellaneous import namedTheta

        self._theta, thetaType = namedTheta(**self._theta)

        self.__is_init = True
        return thetaType

    def parametrization(self, **kwargs):

        sorted_kwargs = {}
        for par in self.parameters:
            if par not in kwargs:
                raise ValueError(par + " is missing in the parametrization tuple")
            else:
                sorted_kwargs[par] = kwargs[par]

        self._theta_parametrization = pc.NamedTuple(**sorted_kwargs)
        return pc.NamedTuple(**sorted_kwargs)

    def add_MCMC(
        self,
        name: str,
        x0: float,
        size: int,
        sd: float,
        mean_name: str,
        variance_name: Optional[str] = None,
    ) -> None:

        if not isinstance(mean_name, str):
            raise TypeError("mean_name must be a str")
        if mean_name not in self.parameters:
            raise KeyError(mean_name + " does not exist in parameters")
        mean = self.parameters[mean_name].data()

        variance = np.array([1])
        if variance_name is None:
            warn(
                "no variance was provided, the default variance was set to 1 as hyper parameter"
            )
        else:
            if not isinstance(variance_name, str):
                raise TypeError("variance_name must be a str")
            if variance_name not in self.parameters:
                raise KeyError(variance_name + "does not exist in parameters")

            variance = self.parameters[variance_name].data()

        self.latent_variables[name] = MCMC_chain(x0, size, sd, mean, variance, name)

    def __repr__(self) -> str:
        msg = "[ == solver === ]\n\t*latent variables :"
        for var in self.latent_variables.values():
            msg += "\n\t\t-" + str(var)

        msg += "\n\t*parameters :"
        for par in self.parameters.values():
            msg += "\n\t\t-" + str(par)

        # msg += "\n\t*theta :" + str(self._theta)

        msg += "\n\t*data : "
        for k in self._data:
            msg += k + ", "

        return msg


def solver_init(s: solver, theta0, mean_name, variance_name, mcmc_name, dim, sd):
    from sdg4varselect.parameter import par_mean, par_variance

    for k in theta0:

        if mean_name in k:
            name_len = len(mean_name)

            mcmc = mcmc_name + k[name_len:]
            variance = variance_name + k[name_len:]

            s.add_parameter(par_mean(theta0[k], mcmc, k))

            if variance in theta0:
                s.add_parameter(par_variance(theta0[variance], mcmc, variance))
                s.add_MCMC(mcmc, theta0[k], dim[mcmc], sd[mcmc], k, variance)
            else:
                s.add_MCMC(mcmc, theta0[k], dim[mcmc], sd[mcmc], k)
    return s
