"""
Module for abstract class AbstractModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

from abc import abstractmethod
import functools

import numpy as np


import jax.numpy as jnp
from jax import jit, jacfwd

import parametrization_cookbook.jax as pc


class AbstractModel:
    """the most abstact model that can be defined"""

    def __init__(self, N, **kwargs):
        self._cst = kwargs
        self._n = N

        self._parametrization: pc.NamedTuple = None

    @abstractmethod
    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""

    @property
    def N(self):
        return self._n

    @property
    @abstractmethod
    def name(self):
        """return a str called name, based on the parameter of the model"""

    @property
    def parametrization_size(self):
        zeros = self._parametrization.reals1d_to_params(
            jnp.zeros(
                shape=self._parametrization.size,
            )
        )
        return self.hstack_params(zeros).shape[0]

    @property
    def parametrization(self):
        return self._parametrization

    @property
    def params_names(self):
        idx_params = self._parametrization.idx_params

        zeros = self._parametrization.reals1d_to_params(
            jnp.zeros(
                shape=self._parametrization.size,
            )
        )
        rep_num = [jnp.array(p).flatten("C").shape[0] for p in zeros]
        # [idx.stop - idx.start for idx in idx_params]
        repeat_name = np.repeat(idx_params._fields, rep_num)

        index_rep = np.concatenate(
            [[""] if n == 1 else [i for i in range(n)] for n in rep_num]
        )
        return np.char.add(repeat_name, index_rep)

    def paramslist_to_reals1d(self, params):
        params_dict = dict(
            [(key, params[item]) for key, item in self.parametrization._idx.items()]
        )

        return self.parametrization.params_to_reals1d(params_dict)

    def hstack_params(self, params):
        return jnp.hstack([jnp.array(p).flatten("C") for p in list(params)])

    def reals1d_to_hstack_params(self, theta_reals1d):
        params = self._parametrization.reals1d_to_params(theta_reals1d)

        return self.hstack_params(params)

    def new_params(self, **kwargs):
        theta_reals1d = self._parametrization.params_to_reals1d(**kwargs)
        return self._parametrization.reals1d_to_params(theta_reals1d)

    # ============================================================== #
    # @abstractmethod
    # @functools.partial(jit, static_argnums=0)
    # def log_likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
    #     """return likelihood with only the gaussian prior"""

    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d, **kwargs):
        """return log likelihood as array each component for each individuals"""

    @functools.partial(jit, static_argnums=0)
    def log_likelihood(self, theta_reals1d, **kwargs):
        return self.log_likelihood_array(theta_reals1d, **kwargs).sum()

    @functools.partial(jit, static_argnums=0)
    def jac_log_likelihood(self, theta_reals1d, **kwargs):
        return jacfwd(self.log_likelihood_array)(theta_reals1d, **kwargs)

    # ============================================================== #
    @abstractmethod
    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ):
        """Sample one data set for the model"""


# def sample_model(prngkey, params_star, model: type(AbstractModel), **kwargs):
#     obs, _ = model.sample(prngkey=prngkey, params_star=params_star, **kwargs)
#     return DataHandler(**obs)
