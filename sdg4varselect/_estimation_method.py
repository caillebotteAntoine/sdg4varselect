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
    **kwargs,
):
    selected_component = (theta_first_estim != 0).at[: model.DIM_LD].set(True)

    # === ESTIMATION === #
    (data_shrink, model_shrink) = AbstractHDModel.shrink_model_and_data(
        model, data, selected_component
    )

    print(f"the model has been shrunk to P = {model_shrink.P}")
    print(data_shrink["cov"].shape)

    # print(
    #     f"p size {model_shrink.parametrization} size {model_shrink.parametrization_size}"
    # )
    # print(f"p size {model_shrink.parametrization.size} P = {model_shrink.P}")
    # print(f"cov shape = {data_shrink['cov'].shape} lbd = {kwargs['lbd'].shape}")

    res_estim = estim_fct(prngkey, model_shrink, data_shrink, **kwargs)

    # === THETA RE CONSTRUCTION === #
    # dim_ld = model.DIM_LD
    # non_zero_component = jnp.concatenate([jnp.repeat(True, dim_ld), hd_selected])
    res_estim.expand_theta(selected_component)
    return res_estim


def lasso_into_estim(
    estim_fct: Callable[
        [jnp.ndarray, type[AbstractModel], dict, int], tuple[type[sdg4vsResults]]
    ],
    prngkey,
    model: type[AbstractHDModel],
    data,
    lbd,
    **kwargs,
):
    """perform first a lasso and an adapative lasso with the previous results
    and finally estim the parameter on the selected component by the adaptative lasso"""
    prngkey_lasso, prngkey_estim = jrd.split(prngkey, 2)
    lasso = estim_fct(prngkey_lasso, model, data, lbd=lbd, **kwargs)

    estim = _estim_shrink_model(
        estim_fct,
        prngkey_estim,
        model,
        data,
        theta_first_estim=lasso.last_theta,
        lbd=None,
        **kwargs,
    )

    return MultiRunRes([lasso, estim])


def lasso_into_adaptive_into_estim(
    estim_fct: Callable[
        [jnp.ndarray, type[AbstractModel], dict, int], tuple[type[sdg4vsResults]]
    ],
    prngkey,
    model: type[AbstractHDModel],
    data,
    lbd,
    **kwargs,
):
    """perform first a lasso and an adapative lasso with the previous results
    and finally estim the parameter on the selected component by the adaptative lasso"""
    prngkey_lasso, prngkey_adaptive, prngkey_estim = jrd.split(prngkey, 3)
    lasso = estim_fct(prngkey_lasso, model, data, lbd=lbd, **kwargs)

    theta = lasso.last_theta
    # lasso_selected_component = jnp.where(model.hd_mask, theta != 0, True)
    # lbd_weighted = lbd / jnp.abs(theta[lasso_selected_component])

    P = model.P
    lasso_selected_component = theta[-P:][theta[-P:] != 0]

    lbd_weighted = 1 / jnp.hstack(
        [
            jnp.ones(shape=(model.parametrization.size - P,), dtype=jnp.bool),
            lasso_selected_component,
        ]
    )

    adaptive_lasso = lasso
    # adaptive_lasso = _estim_shrink_model(
    #     estim_fct,
    #     prngkey_adaptive,
    #     model,
    #     data,
    #     theta_first_estim=lasso.last_theta,
    #     lbd=lbd_weighted,
    #     **kwargs,
    # )

    estim = _estim_shrink_model(
        estim_fct,
        prngkey_estim,
        model,
        data,
        theta_first_estim=adaptive_lasso.last_theta,
        lbd=None,
        **kwargs,
    )

    return MultiRunRes([lasso, adaptive_lasso, estim])
