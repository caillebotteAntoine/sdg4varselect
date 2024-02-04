"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.outputs import SDGResults
from sdg4varselect.models.pharmacokinetic import PharmaJM as model_jm

from results.logistic_model.one_estim import one_estim


def _estim_shrink_model(
    prngkey, model, dh, selected_component, lbd=None, save_all=True
):
    # === MODEL SHRINKAGE === #
    dim_ld = model.DIM_LD
    hd_selected = selected_component[dim_ld:]
    new_dim_hd = int(hd_selected.sum())

    model_shrink = model_jm(N=model.N, J=model.J, DIM_HD=new_dim_hd)
    dh_shrink = dh.deepcopy()
    dh_shrink.data["cov"] = dh.data["cov"][:, hd_selected]

    # === ESTIMATION === #
    res_estim = one_estim(prngkey, model_shrink, dh_shrink, lbd=lbd, save_all=save_all)

    # === THETA RE CONSTRUCTION === #
    non_zero_component = jnp.concatenate([jnp.repeat(True, dim_ld), hd_selected])

    return SDGResults.expand_theta(res_estim, non_zero_component)


def lasso_into_adaptive_into_estim(prngkey, model, dh, lbd, save_all):
    prngkey_lasso, prngkey_adaptive, prngkey_estim = jrd.split(prngkey, 3)
    lasso = one_estim(prngkey_lasso, model, dh, lbd=lbd, save_all=save_all)

    theta = lasso.theta[-1]
    lasso_selected_component = theta != 0
    lbd_weighted = lbd / jnp.abs(theta[lasso_selected_component])

    adaptive_lasso = _estim_shrink_model(
        prngkey_adaptive,
        model,
        dh,
        selected_component=lasso_selected_component,
        lbd=lbd_weighted,
        save_all=save_all,
    )

    estim = _estim_shrink_model(
        prngkey_estim,
        model,
        dh,
        selected_component=adaptive_lasso.last_theta != 0,
        lbd=None,
        save_all=save_all,
    )

    return [lasso, adaptive_lasso, estim]


if __name__ == "__main__":
    from sdg4varselect import sample_model
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.pharmacokinetic import get_params_star

    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=5)
    myModel = model_jm(N=100, J=5, DIM_HD=10)
    params_star = get_params_star(myModel.DIM_HD)

    myDH = sample_model(
        jrd.PRNGKey(1), params_star, myModel, weibull_censoring_loc=2000
    )

    lasso_r, adaptive_lasso_r, estim_r = lasso_into_adaptive_into_estim(
        jrd.PRNGKey(10), myModel, myDH, 10**-1, save_all=True
    )

    # === PLOT === #
    params_names = myModel.params_names

    # sdgplt.plot_theta(lasso, myModel.DIM_LD, params_star, params_names)
    sdgplt.plot_theta_HD(lasso_r, myModel.DIM_LD, params_star, params_names)

    # sdgplt.plot_theta(adaptive_lasso, myModel.DIM_LD, params_star, params_names)
    sdgplt.plot_theta_HD(adaptive_lasso_r, myModel.DIM_LD, params_star, params_names)

    sdgplt.plot_theta(estim_r, myModel.DIM_LD, params_star, params_names)
    sdgplt.plot_theta_HD(estim_r, myModel.DIM_LD, params_star, params_names)
