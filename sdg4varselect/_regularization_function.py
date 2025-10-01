"""
Module for Regularization Path and Gradient Descent Optimization in High-Dimensional Models.

Created by antoine.caillebotte@inrae.fr
"""

from typing import Callable
import functools

import datetime

import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.models import AbstractHDModel
from sdg4varselect.outputs import Sdg4vsResults, RegularizationPath


def _add_flag(estim_fct) -> Callable[[], tuple[type[Sdg4vsResults], bool]]:
    """Decorate an estimation function to include a completion flag.

    This decorator modifies the estimation function to return a tuple containing
    the estimation results and a flag that indicates whether the regularization path
    has completed (i.e., if no non-zero parameters remain in the final parameter set).

    Parameters
    ----------
    estim_fct : Callable
        The estimation function to be decorated.

    Returns
    -------
    Callable
        A wrapped version of the estimation function that returns the results along
        with a completion flag.
    """

    @functools.wraps(estim_fct)
    def estim_fct_with_flag(  # pylint: disable=missing-return-doc
        model: type[AbstractHDModel], **kwargs
    ) -> tuple[Sdg4vsResults, bool]:
        res_estim = estim_fct(model=model, **kwargs)
        P = model.P
        flag = (res_estim.last_theta[-P:] != 0).sum() == 0

        return res_estim, flag

    return estim_fct_with_flag


def regularization_path(
    estim_fct: Callable[[], type[Sdg4vsResults]],
    prngkey,
    lbd_set: jnp.ndarray,
    model: type[AbstractHDModel],
    **kwargs
) -> RegularizationPath:
    """Perform a regularization path using a specified estimation function.

    This function iterates over a set of regularization parameters (`lbd_set`)
    and applies the estimation function to the model for each parameter. The process
    stops if the selection flag indicates that no further regularization steps are
    needed.

    Parameters
    ----------
    estim_fct : Callable
        The estimation function to apply at each step of the regularization path.
    prngkey : jnp.ndarray
        The random key for the estimation function, compatible with JAX's random keys.
    lbd_set : jnp.ndarray
        The set of regularization penalty values to iterate over.
    model : type[AbstractHDModel]
        The high-dimensional model for which the regularization path is computed.
    **kwargs : dict
        Additional arguments passed to the estimation function.

    Returns
    -------
    RegularizationPath
        An instance of `RegularizationPath` containing the results across the regularization path.
    """
    prngkey_list = jrd.split(prngkey, num=len(lbd_set))
    assert isinstance(model, AbstractHDModel), "the model must be an HDModel"

    estim_fct_with_flag = _add_flag(estim_fct)

    def iter_estim():  # pylint: disable=W9013,W9014
        for i, lbd in enumerate(lbd_set):
            kwargs["lbd"] = lbd
            kwargs["prngkey"] = prngkey_list[i]
            res_estim, flag_selection = estim_fct_with_flag(model=model, **kwargs)

            if flag_selection:
                yield res_estim
                res_estim.chrono = datetime.timedelta(0)
                for _ in range(len(lbd_set) - i - 1):
                    yield res_estim
                break
            yield res_estim

    out = RegularizationPath(results=list(iter_estim()), lbd_set=lbd_set)
    out.update_bic(model)
    return out
