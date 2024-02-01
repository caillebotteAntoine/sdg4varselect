"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import regularization_path
from sdg4varselect.outputs import VariableSelectionRes
import sdg4varselect.models.logistic_joint_model as modelisation

from results.logistic_model.one_selection_and_estimation import (
    lasso_into_adaptive_into_estim,
)


def estim_with_flag(model, *args, **kwargs):
    """must return the estimation results and
    a flag which indicates if the regularization path is finished"""
    res_estim = lasso_into_adaptive_into_estim(*args, model=model, **kwargs)
    dim_ld = model.DIM_LD
    flag = (res_estim[-1].last_theta[dim_ld:] != 0).sum() == 0

    return res_estim, flag


def _one_result(prngkey, model, dh, lbd_set, save_all=True):
    reg_path_res = regularization_path(
        estim_fct_with_flag=estim_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        dim_ld=model.DIM_LD,
        N=model.N * (1 + model.J),
        verbatim=True,  # __name__ == "__main__",
        # additional parameter
        model=model,
        dh=dh,
        save_all=save_all,
    )

    reg_path = [res[-1] for res in reg_path_res.listSDGResults]
    argmin_bic = reg_path_res.bic[-1].argmin()

    return VariableSelectionRes(
        listSDGResults=reg_path_res.listSDGResults,
        theta=reg_path[argmin_bic],
        argmin_bic=argmin_bic,
        bic=reg_path_res.bic,
        regularization_path=reg_path,
    )


def one_result(args):
    prngkey, N, J, DIM_HD, dh, lbd_set, save_all = args

    model = modelisation.Logistic_JM(N, J, DIM_HD)
    return _one_result(prngkey, model, dh, lbd_set, save_all)


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt

    my_lbd_set = 10 ** jnp.linspace(-2, -0.5, num=15)
    myModel = modelisation.Logistic_JM(N=100, J=5, DIM_HD=10)

    myDH = modelisation.sample_one(jrd.PRNGKey(10), myModel, weibull_censoring_loc=2000)

    sres = _one_result(jrd.PRNGKey(10), myModel, myDH, my_lbd_set, save_all=True)

    # === PLOT === #

    params_star = modelisation.get_params_star(myModel.DIM_HD)

    # sdgplt.plot_theta(
    #     sres.listSDGResults[-1], myModel.DIM_LD, params_star, myModel.params_names
    # )
    sdgplt.plot_theta_HD(
        sres.listSDGResults[sres.argmin_bic][-1],
        myModel.DIM_LD,
        params_star,
        myModel.params_names,
    )

    sdgplt.plot_reg_path(
        my_lbd_set,
        [res[-1] for res in sres.listSDGResults],
        sres.bic[-1],
        myModel.DIM_LD,
    )
