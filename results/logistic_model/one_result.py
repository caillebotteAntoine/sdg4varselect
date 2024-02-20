"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import sdgplt, regularization_path, lasso_into_adaptive_into_estim
from sdg4varselect.outputs import RegularizationPathRes, MultiRunRes

from sdg4varselect.models import create_cox_mem_jm, logisticMEM

from results.logistic_model.one_estim import one_estim


def estim_with_flag(model, **kwargs) -> tuple[MultiRunRes, bool]:
    """must return the estimation results and
    a flag which indicates if the regularization path is finished"""
    res_estim = lasso_into_adaptive_into_estim(one_estim, model=model, **kwargs)
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
    model = create_cox_mem_jm(logisticMEM, N, J, P)

    return _one_result(prngkey, model, data, lbd_set, save_all)


if __name__ == "__main__":
    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=15)
    my_lbd_set = [2 * 10**-1]

    myModel = create_cox_mem_jm(logisticMEM, 100, 5, 10)
    p_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=110.1,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, _ = myModel.sample(p_star, jrd.PRNGKey(10), weibull_censoring_loc=77)

    res = _one_result(jrd.PRNGKey(0), myModel, myobs, my_lbd_set, save_all=True)

    # === PLOT === #
    params_names = myModel.params_names

    sdgplt.plot_theta(res, 7, p_star, params_names)
    sdgplt.plot_theta_hd(res, 7, p_star, params_names)
    sdgplt.plot_reg_path(res, myModel.DIM_LD)
    print(f"chrono = {res.chrono}")
