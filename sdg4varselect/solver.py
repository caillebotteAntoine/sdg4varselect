# Create by caillebotte.antoine@inrae.fr

import jax.numpy as jnp
import numpy as np
import scipy.special
import parametrization_cookbook.jax as pc

from sdg4varselect.MCMC import MCMC_chain

import jax.random as jrd
from time import time


class Solver:
    def __init__(self):
        """Constructor of solver."""
        self._theta_reals1d: jnp.ndarray = None
        self._theta0_reals1d: jnp.ndarray = None
        self._parametrization: pc.NamedTuple = None

        self.latent_variables: dict[str, MCMC_chain] = {}
        self.__data = {}

        self._likelihood = None
        self._likelihood_kwargs = {}

        self.__is_init = False

    def is_init(self):
        """returns true if the initial parameterization and parameters were provided and initialized"""
        return self.__is_init

    @property
    def parametrization(self) -> pc._common.NamedTuple:
        """returns theta parametrization"""
        return self._parametrization

    @parametrization.setter
    def parametrization(self, para):  # kwargs):
        """Define the parametrization for theta"""
        self._parametrization = para  # pc.NamedTuple(**kwargs)

    @property
    def params(self):
        """returns the last estimated value of the parameters"""
        return self._parametrization.reals1d_to_params(self._theta_reals1d)

    def parametrization_reals1d_to_params(self, x):
        return self._parametrization.reals1d_to_params(x)

    @property
    def theta_reals1d(self):
        """returns theta in the reparameterized space"""
        return self._theta_reals1d

    @theta_reals1d.setter
    def theta_reals1d(self, kwargs):
        if self._parametrization is None:
            raise ValueError("parametrization must be initiate first.")

        if isinstance(kwargs, dict):
            self._theta_reals1d = self._parametrization.params_to_reals1d(**kwargs)
        else:
            self._theta_reals1d = self._parametrization.params_to_reals1d(kwargs)

        self.__is_init = True

    @property
    def params_names(self):
        idx_params = self.parametrization.idx_params
        rep_num = [idx.stop - idx.start for idx in idx_params]
        names = idx_params._fields
        return np.repeat(names, rep_num)

    @property
    def likelihood(self):
        """returns the likelihood"""
        return self._likelihood

    @likelihood.setter
    def likelihood(self, func):
        self._likelihood = func

    @property
    def likelihood_kwargs(self) -> dict:
        """returns the likelihood kwargs"""
        return self._likelihood_kwargs

    def add_likelihood_kwargs(self, *name_args) -> None:
        # self.set_likelihood_kwargs(*name_args)

        # def set_likelihood_kwargs(self, *name_args) -> None:
        """define the default parameters of the function to be optimized by
        looking for them in the latent variables or the data"""
        for name in name_args:
            if isinstance(name, str):
                if name in self.latent_variables:
                    self._likelihood_kwargs[name] = self.latent_variables[name].data
                elif name in self.__data:
                    self._likelihood_kwargs[name] = self.__data[name]
                else:
                    raise KeyError(
                        f"{name} does not exist neiter in latent variables or in global variables."
                    )
            else:
                raise TypeError("name must be a str or a list of str")

    def add_data(self, **kwargs) -> None:
        """adds variables to the solver data"""
        for key, item in kwargs.items():
            if key in self.__data:
                raise KeyError(key + " all ready exist in solver's data.")
            self.__data[key] = item

    def update_data(self, **kwargs) -> None:
        """update variables to the solver data"""
        for key, item in kwargs.items():
            if key in self.latent_variables:
                raise KeyError(
                    f"changing the value of a latent variable ({key}) is not allowed."
                )

            if key in self._likelihood_kwargs:
                self._likelihood_kwargs[key] = item

            if key in self.__data:
                self.__data[key] = item
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

    def __repr__(self) -> str:
        msg = "[ == solver === ]\n\t*latent variables :"
        for var in self.latent_variables.values():
            msg += "\n\t\t-" + str(var)

        msg += (
            str(self.params)
            .replace("Array", "")
            .replace(", dtype=float32", "")
            .replace(", weak_type=True", "")
            .replace("Parameters(", "\n\t*parameters :\n\t\t")
            .replace(", ", ",\n\t\t")
        )

        # msg += "\n\t*theta :" + str(self._theta)

        msg += "\n\t*data : "
        for k in self.__data:
            msg += k + ", "

        msg += "\n\t*likelihood_kwargs : "
        for k in self._likelihood_kwargs:
            msg += k + ", "

        return msg

    def likelihood_marginal(self, size, theta=None):
        if theta is None:
            theta = self.theta_reals1d

        var_lat_sample = {}

        for var in self.latent_variables:
            var_lat_sample[var] = self.latent_variables[var].sample(
                jrd.PRNGKey(int(time())),
                theta,
                size=size - 1,
                **self.likelihood_kwargs,
            )

        out = self.likelihood(theta, **self.likelihood_kwargs)
        for k in range(size - 1):
            for var in self.latent_variables:
                self.likelihood_kwargs[var] = var_lat_sample[var][k]

            out += self.likelihood(theta, **self.likelihood_kwargs)

        for var in self.latent_variables:
            self.likelihood_kwargs[var] = self.latent_variables[var].data
        return out / size

    def theta_nonzero_support(self, p, theta=None):
        """return the mask of non zero last p component of theta"""
        if theta is None:
            theta = self.theta_reals1d

        d = len(theta) - p
        LD_mask = [True for k in range(d)]
        HD_mask = jnp.arange(len(theta)) >= d
        return jnp.hstack([LD_mask, theta[HD_mask] != 0])

    def get_number_of_nonzero(self, p):
        d = len(self.theta_reals1d) - p
        return self.theta_nonzero_support(p=p).sum() - d

    def shrink_theta(self, index):
        self._theta_reals1d = self._theta_reals1d[index]

    def BIC(self, n, p, theta=None, size=100):
        """
        BIC = k*ln(n) - 2*ln(L)

        where :
            - k is the number of parameter estimated (ie non zero parameter in HD parameter)
            - n is the sample size
            - L the maximzed value of the likelihood function
        """
        if theta is None:
            theta = self.theta_reals1d

        k = jnp.count_nonzero(self.theta_nonzero_support(p, theta=theta))
        log_L = self.likelihood_marginal(size=size, theta=theta)

        return -2 * log_L + k * jnp.log(n)

    def eBIC(self, n, p, theta=None, size=100):
        """
        eBIC = k*ln(n) - 2*ln(L) + 2*ln(C^p_k)

        where :
            - k is the number of parameter estimated (ie non zero parameter in HD parameter)
            - n is the sample size
            - L the maximzed value of the likelihood function
        """

        k = self.get_number_of_nonzero(p=p)
        ebic_pen = scipy.special.binom(p, k)

        return self.BIC(n, p, theta=theta, size=size) + ebic_pen


def shrink_support(solver, name, p):
    from copy import deepcopy

    solver_shrink = deepcopy(solver)
    solver_shrink.parametrization = deepcopy(solver.parametrization)
    d = len(solver.theta_reals1d) - p

    # COV shrinkage
    mask_select = solver_shrink.theta_nonzero_support(p=p)
    # id_select = np.where(mask_select)[0]
    cov = deepcopy(solver_shrink.likelihood_kwargs["cov"])
    cov_shrink = jnp.where(mask_select[d:], cov, 0)
    # cov[:, id_select[d:]]
    # On met des zeros plutot que de reparametriser

    solver_shrink.update_data(cov=cov_shrink)

    return solver_shrink, mask_select
