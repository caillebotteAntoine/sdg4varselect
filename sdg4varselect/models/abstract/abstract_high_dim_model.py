"""
Module for the AbstractHDModel class.

This module defines the `AbstractHDModel` class, which serves as a base class for high-dimensional models,
establishing fundamental properties and methods for model parameter handling and high-dimensional masking.

Created by antoine.caillebotte@inrae.fr
"""

import jax.numpy as jnp


class AbstractHDModel:
    """Base class for high-dimensional models.

    Parameters
    ----------
    P : int
        Number of high-dimensional parameters in the model.

    Attributes
    ----------
    parametrization_size : int
        Total size of the parameter array, including both low- and high-dimensional parameters.
    hd_mask : jnp.ndarray
        Boolean mask indicating which parameters are high-dimensional.
    """

    def __init__(self, P):
        self._p: int = P
        self._parametrization_size: int = None

    @property
    def parametrization_size(self) -> int:
        """Get the size of the parameter array.

        Returns
        -------
        int
            Size of the parameter array, including both low- and high-dimensional parameters.
        """
        return self._parametrization_size

    @parametrization_size.setter
    def parametrization_size(self, size):
        self._parametrization_size = size

    @property
    def P(self) -> int:
        """Get the number of high-dimensional parameters.

        Returns
        -------
        int
            Number of high-dimensional parameters in the model.
        """
        return self._p

    @P.setter
    def P(self, P):
        self._p = int(P)

    @property
    def hd_mask(self) -> jnp.ndarray:
        """Generate a mask indicating high-dimensional parameters.

        Returns
        -------
        jnp.ndarray
            A boolean array where elements corresponding to high-dimensional parameters are set to True.

        Raises
        ------
        AssertionError
            If the parametrization size has not been initialized or if the number of high-dimensional
            parameters exceeds the parametrization size.
        """
        assert (
            self.parametrization_size is not None
        ), "parametrization_size has not been updated in init_parametrization!"
        assert (
            self.P <= self.parametrization_size
        ), "high dimension length must be smaller than the parametrization_size"
        return (
            jnp.arange(self.parametrization_size) >= self.parametrization_size - self.P
        )

    # @classmethod
    # def shrink_model_and_data(
    #     cls, model, data, selected_component, data_to_shrink=["cov"]
    # ):

    #     # === MODEL SHRINKAGE === #
    #     P = model.P
    #     hd_selected = selected_component[-P:]
    #     new_dim_hd = int(hd_selected.sum())

    #     model_shrink = deepcopy(model)
    #     model_shrink.P = new_dim_hd
    #     model_shrink.init()

    #     data_shrink = data.copy()
    #     for key in data_to_shrink:
    #         data_shrink["cov"] = data_shrink["cov"][:, hd_selected]

    #     return (data_shrink, model_shrink)
