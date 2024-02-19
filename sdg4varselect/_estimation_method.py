from typing import Callable


import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.outputs import sdg4vsResults, GDResults, MultiRunRes
from sdg4varselect.models.abstract.abstract_high_dim_model import AbstractHDModel
from sdg4varselect.models.abstract.abstract_model import AbstractModel


def _estim_shrink_model(
    estim_fct: Callable[[], tuple[type[sdg4vsResults]]],
    prngkey,
    model: type[AbstractHDModel],
    data,
    theta_first_estim,
    **kwargs
):
    selected_component = theta_first_estim != 0

    # === ESTIMATION === #
    (data_shrink, model_shrink) = AbstractHDModel.shrink_model_and_data(
        model, data, selected_component
    )

    res_estim = estim_fct(prngkey, model_shrink, data_shrink, **kwargs)

    # === THETA RE CONSTRUCTION === #
    # dim_ld = model.DIM_LD
    # non_zero_component = jnp.concatenate([jnp.repeat(True, dim_ld), hd_selected])

    return GDResults.expand_theta(res_estim, selected_component)


def lasso_into_adaptive_into_estim(
    estim_fct: Callable[
        [jrd.PRNGKeyArray, type[AbstractModel], dict, int], tuple[type[sdg4vsResults]]
    ],
    prngkey,
    model: type[AbstractHDModel],
    data,
    lbd,
    **kwargs
):
    """perform first a lasso and an adapative lasso with the previous results
    and finally estim the parameter on the selected component by the adaptative lasso"""
    prngkey_lasso, prngkey_adaptive, prngkey_estim = jrd.split(prngkey, 3)
    lasso = estim_fct(prngkey_lasso, model, data, lbd=lbd, **kwargs)

    theta = lasso.theta[-1]
    lasso_selected_component = theta != 0
    lbd_weighted = lbd / jnp.abs(theta[lasso_selected_component])

    adaptive_lasso = _estim_shrink_model(
        estim_fct,
        prngkey_adaptive,
        model,
        data,
        theta_first_estim=lasso.last_theta,
        lbd=lbd_weighted,
        **kwargs
    )

    estim = _estim_shrink_model(
        estim_fct,
        prngkey_estim,
        model,
        data,
        theta_first_estim=adaptive_lasso.last_theta,
        lbd=None,
        **kwargs
    )

    return MultiRunRes([lasso, adaptive_lasso, estim])
