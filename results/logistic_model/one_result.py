"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import regularization_path
from sdg4varselect.outputs import RegularizationPathRes, MultiRunRes

from sdg4varselect.models.wcox_mem_joint_model import (
    create_logistic_weibull_jm,
)


from results.logistic_model.one_selection_and_estimation import (
    lasso_into_adaptive_into_estim,
)


def estim_with_flag(model, *args, **kwargs) -> tuple[MultiRunRes, bool]:
    """must return the estimation results and
    a flag which indicates if the regularization path is finished"""
    res_estim = lasso_into_adaptive_into_estim(*args, model=model, **kwargs)
    dim_ld = model.DIM_LD
    flag = (res_estim[-1].last_theta[dim_ld:] != 0).sum() == 0

    return res_estim, flag


def _one_result(prngkey, model, data, lbd_set, save_all=True):

    list_sdg_results, bic = regularization_path(
        estim_fct_with_flag=estim_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        dim_ld=model.DIM_LD,
        N=model.N * (1 + model.J),
        verbatim=True,  # __name__ == "__main__",
        # additional parameter
        model=model,
        data=data,
        save_all=save_all,
    )

    argmin_bic = bic[-1].argmin()

    return RegularizationPathRes(
        multi_run=list_sdg_results,
        argmin_bic=argmin_bic,
        bic=bic,
        lbd_set=lbd_set,
    )


def one_result(args):
    prngkey, N, J, P, data, lbd_set, save_all = args
    model = create_logistic_weibull_jm(N, J, P)

    return _one_result(prngkey, model, data, lbd_set, save_all)


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.wcox_mem_joint_model import (
        get_params_star,
    )

    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=15)
    myModel = create_logistic_weibull_jm(100, 5, 10)
    my_params_star = get_params_star(myModel)

    myobs, _ = myModel.sample(my_params_star, jrd.PRNGKey(10), weibull_censoring_loc=77)

    sres = _one_result(jrd.PRNGKey(0), myModel, myobs, my_lbd_set, save_all=True)

    # === PLOT === #

    # sdgplt.plot_theta(
    #     sres.listSDGResults[-1], myModel.DIM_LD, params_star, myModel.params_names
    # )
    sdgplt.plot_theta_hd(
        sres.multi_run[sres.argmin_bic][-1],
        myModel.DIM_LD,
        my_params_star,
        myModel.params_names,
    )

    sdgplt.plot_reg_path(sres, myModel.DIM_LD)
    print(f"chrono = {sres.chrono}")

    x = RegularizationPathRes.switch_runs(sres)
