"""
Module for abstract class AbstractModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

from abc import ABC, abstractmethod
import functools

# import numpy as np


import jax.numpy as jnp
from jax import jit, jacfwd

import parametrization_cookbook.jax as pc

from sdg4varselect.exceptions import sdg4vsException


def _check_initialization(fun):

    @functools.wraps(fun)
    def new_fun(self, *args, **kwargs):
        if not self.is_initialized:
            raise sdg4vsException(
                "The model has not been initiated and therefore cannot be used !"
            )
        return fun(self, *args, **kwargs)

    return new_fun


class AbstractModel(ABC):
    """the most abstact model that can be defined"""

    def __init__(self, N: int, **kwargs):
        self._is_initialized = False
        self._cst = kwargs
        self._n = N

        self._parametrization: pc.NamedTuple = None

        self.init()

    @abstractmethod
    def init_parametrization(self):
        """here you define the parametrization of the model"""
        raise NotImplementedError("You must define parametrization in this function !")

    def init(self):
        """don't forget to call the mother init function at the end

        After calling this method is_initialized should be True and model should be ready for use
        """
        self.init_parametrization()
        self._is_initialized = True

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    @_check_initialization
    def N(self):
        return self._n

    @property
    @abstractmethod
    @_check_initialization
    def name(self):
        """return a str called name, based on the parameter of the model"""
        raise NotImplementedError

    # @property
    # @_check_initialization
    # def parametrization_size(self):
    #     zeros = self._parametrization.reals1d_to_params(
    #         jnp.zeros(
    #             shape=self._parametrization.size,
    #         )
    #     )
    #     return self.hstack_params(zeros).shape[0]

    @property
    @_check_initialization
    def parametrization(self):
        return self._parametrization

    # @property
    # def params_names(self):
    #     def extract_name(p: pc.NamedTuple):
    #         repeat_name = np.array([])

    #         idx_params = p.idx_params
    #         for name in idx_params._fields:
    #             if isinstance(p[name], pc.NamedTuple):
    #                 repeat_name = np.concatenate([repeat_name, extract_name(p[name])])
    #             else:
    #                 shape = (
    #                     p[name]
    #                     .reals1d_to_params(
    #                         jnp.zeros(
    #                             shape=p[name].size,
    #                         )
    #                     )
    #                     .flatten("C")
    #                     .shape
    #                 )
    #                 if len(shape) == 0 or shape[0] == 1:
    #                     repeat_name = np.concatenate([repeat_name, np.array([name])])
    #                 else:
    #                     repeat_name = np.concatenate(
    #                         [
    #                             repeat_name,
    #                             np.array([f"{name}{i+1}" for i in range(shape[0])]),
    #                         ]
    #                     )

    #         return np.array(repeat_name)

    #     return extract_name(self._parametrization)

    def paramslist_to_reals1d(self, params):
        params_dict = dict(
            [(key, params[item]) for key, item in self.parametrization._idx.items()]
        )

        return self.parametrization.params_to_reals1d(params_dict)

    def hstack_params(self, params):
        return jnp.hstack([jnp.array(p).flatten("C") for p in list(params)])

    # def reals1d_to_hstack_params(self, theta_reals1d):
    #     params = self._parametrization.reals1d_to_params(theta_reals1d)
    #     return self.hstack_params(params)

    @_check_initialization
    def new_params(self, **kwargs):
        theta_reals1d = self._parametrization.params_to_reals1d(**kwargs)
        return self._parametrization.reals1d_to_params(theta_reals1d)

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d, **kwargs):
        """return log likelihood as array each component for each individuals"""
        raise NotImplementedError

    @functools.partial(jit, static_argnums=0)
    def log_likelihood(self, theta_reals1d, **kwargs):
        return self.log_likelihood_array(theta_reals1d, **kwargs).sum()

    @functools.partial(jit, static_argnums=0)
    def jac_log_likelihood(self, theta_reals1d, **kwargs):
        return jacfwd(self.log_likelihood_array)(theta_reals1d, **kwargs)

    # ============================================================== #
    @abstractmethod
    @_check_initialization
    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ):
        """Sample one data set for the model"""
        raise NotImplementedError


if __name__ == "__main__":
    m = AbstractModel(0)
    m.name
