"""
Module for abstract class AbstractHDModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

from copy import deepcopy
import jax.numpy as jnp


class AbstractHDModel:
    """the most abstact model with high dim that can be defined"""

    def __init__(self, P, **kwargs):
        self._p = P
        # self._dim = None

    # def init_dim(self):
    #     self._dim = self.parametrization_size

    # @property
    # def DIM_LD(self):
    #     return self._dim - self.P

    @property
    def P(self):
        return self._p

    @P.setter
    def P(self, P):
        self._p = P

    # @property
    # def hd_mask(self):
    #     return jnp.arange(self.parametrization.size) >= self.DIM_LD

    @classmethod
    def shrink_model_and_data(cls, model, data, selected_component):

        # === MODEL SHRINKAGE === #
        P = model.P
        hd_selected = selected_component[-P:]
        new_dim_hd = int(hd_selected.sum())

        model_shrink = deepcopy(model)
        model_shrink.P = new_dim_hd
        model_shrink.init()

        data_shrink = data.copy()
        data_shrink["cov"] = data_shrink["cov"][:, hd_selected]

        return (data_shrink, model_shrink)
