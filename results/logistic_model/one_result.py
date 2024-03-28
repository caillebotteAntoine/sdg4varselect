"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import sdgplt, regularization_path, lasso_into_adaptive_into_estim
from sdg4varselect.outputs import RegularizationPathRes, MultiRunRes

from sdg4varselect.models import WeibullCoxJM, logisticMEM

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

    model = WeibullCoxJM(logisticMEM(N=N, J=J), P=P, alpha_scale=0.001, a=800, b=10)

    return _one_result(prngkey, model, data, lbd_set, save_all)


if __name__ == "__main__":
    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=5)
    myModel = WeibullCoxJM(
        logisticMEM(N=500, J=15), P=50, alpha_scale=0.001, a=800, b=10
    )

    print(f"Estimation of computation time {myModel.P*0.023}")

    p_star = myModel.new_params(
        mean_latent={"mu1": 200, "mu2": 500},
        mu3=150,
        cov_latent=jnp.diag(jnp.array([40, 100])),
        var_residual=100,
        alpha=0.005,
        beta=jnp.concatenate(  # jnp.zeros(shape=(myModel.P,)),  #
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, _ = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=7700)

    res = _one_result(jrd.PRNGKey(0), myModel, myobs, my_lbd_set, save_all=True)

    # === PLOT === #
    params_names = myModel.params_names

    sdgplt.plot_theta(res, 9, myModel.hstack_params(p_star), params_names)
    sdgplt.plot_theta_hd(res, 9, myModel.hstack_params(p_star), params_names)
    sdgplt.plot_reg_path(res, myModel.DIM_LD)
    print(f"chrono = {res.chrono}")
