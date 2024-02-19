"""
Module for abstract class AbstractHDModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116

from copy import deepcopy


class AbstractHDModel:
    """the most abstact model with high dim that can be defined"""

    def __init__(self, P, **kwargs):
        self._p = P
        self._dim = None

    def init_dim(self, dim):
        self._dim = dim

    @property
    def DIM_LD(self):
        return self._dim - self.P

    @property
    def P(self):
        return self._p

    @P.setter
    def P(self, P):
        self._p = P

    @classmethod
    def shrink_model_and_data(cls, model, data, selected_component):

        # === MODEL SHRINKAGE === #
        dim_ld = model.DIM_LD
        hd_selected = selected_component[dim_ld:]
        new_dim_hd = int(hd_selected.sum())

        model_shrink = deepcopy(model)
        model_shrink.P = new_dim_hd
        model_shrink.init()

        data_shrink = data.copy()
        data_shrink["cov"] = data_shrink["cov"][:, hd_selected]

        return (data_shrink, model_shrink)
