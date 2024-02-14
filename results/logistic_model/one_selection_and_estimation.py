"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.outputs import GDResults, MultiRunRes

from sdg4varselect.models.wcox_mem_joint_model import (
    create_logistic_weibull_jm as modelisation,
)


from results.logistic_model.one_estim import one_estim


def _estim_shrink_model(
    prngkey, model, data, selected_component, lbd=None, save_all=True
):
    # === MODEL SHRINKAGE === #
    dim_ld = model.DIM_LD
    hd_selected = selected_component[dim_ld:]
    new_dim_hd = int(hd_selected.sum())

    model_shrink = modelisation(N=model.N, J=model.J, P=new_dim_hd)
    data_shrink = data.copy()
    data_shrink["cov"] = data["cov"][:, hd_selected]

    # === ESTIMATION === #
    res_estim = one_estim(
        prngkey, model_shrink, data_shrink, lbd=lbd, save_all=save_all
    )

    # === THETA RE CONSTRUCTION === #
    non_zero_component = jnp.concatenate([jnp.repeat(True, dim_ld), hd_selected])

    return GDResults.expand_theta(res_estim, non_zero_component)


def lasso_into_adaptive_into_estim(prngkey, model, data, lbd, save_all):
    prngkey_lasso, prngkey_adaptive, prngkey_estim = jrd.split(prngkey, 3)
    lasso = one_estim(prngkey_lasso, model, data, lbd=lbd, save_all=save_all)

    theta = lasso.theta[-1]
    lasso_selected_component = theta != 0
    lbd_weighted = lbd / jnp.abs(theta[lasso_selected_component])

    adaptive_lasso = _estim_shrink_model(
        prngkey_adaptive,
        model,
        data,
        selected_component=lasso_selected_component,
        lbd=lbd_weighted,
        save_all=save_all,
    )

    estim = _estim_shrink_model(
        prngkey_estim,
        model,
        data,
        selected_component=adaptive_lasso.last_theta != 0,
        lbd=None,
        save_all=save_all,
    )

    return MultiRunRes([lasso, adaptive_lasso, estim])


if __name__ == "__main__":
    from sdg4varselect.models.wcox_mem_joint_model import (
        get_params_star,
    )

    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=1)
    myModel = modelisation(100, 5, 10)
    my_params_star = get_params_star(myModel)

    myobs, _ = myModel.sample(my_params_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

    multi_estim = lasso_into_adaptive_into_estim(
        jrd.PRNGKey(0), myModel, myobs, 10**-1, save_all=True
    )

    print(multi_estim.chrono)

    # === PLOT === #
    from sdg4varselect.plot import (
        plot_theta,
        plot_theta_hd,
    )

    params_names = myModel.params_names

    plot_theta(multi_estim, 7, my_params_star, params_names)
    plot_theta_hd(multi_estim, 7, my_params_star, params_names)

    # # === PLOT === #
    # params_names = myModel.params_names

    # # sdgplt.plot_theta(lasso, myModel.DIM_LD, params_star, params_names)
    # sdgplt.plot_theta_HD(lasso_r, myModel.DIM_LD, my_params_star, params_names)

    # # sdgplt.plot_theta(adaptive_lasso, myModel.DIM_LD, params_star, params_names)
    # sdgplt.plot_theta_HD(adaptive_lasso_r, myModel.DIM_LD, my_params_star, params_names)

    # sdgplt.plot_theta(estim_r, myModel.DIM_LD, my_params_star, params_names)
    # sdgplt.plot_theta_HD(estim_r, myModel.DIM_LD, my_params_star, params_names)
