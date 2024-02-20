"""
Module that define functions to perform a simple estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import sdgplt, regularization_path, lasso_into_adaptive_into_estim
from sdg4varselect.outputs import GDResults, RegularizationPathRes, MultiRunRes
from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings

from sdg4varselect.models import create_cox_mem_jm, pkMEM

algo_settings = get_GDFIM_settings(preheating=1000, heating=1400)


def one_estim(prngkey, model, data, lbd=None, alpha=1.0, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization_size,))
    params0 = model.parametrization.reals1d_to_params(theta0)

    algo = SPGD_FIM(prngkey_estim, 2000, algo_settings, lbd=lbd, alpha=alpha)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(params0.mu1),
        sd=0.1,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi1",
    )
    algo.latent_variables["phi1"].adaptative_sd = True
    algo.add_mcmc(
        float(params0.mu2),
        sd=0.1,
        size=model.N,
        likelihood=model.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=True)

    # for var in algo.latent_variables.values():
    #     sdgplt.plot(var)

    return res if save_all else GDResults.make_it_lighter(res)


def estim_with_flag(model, **kwargs) -> tuple[MultiRunRes, bool]:
    """must return the estimation results and
    a flag which indicates if the regularization path is finished"""
    res_estim = lasso_into_adaptive_into_estim(one_estim, model=model, **kwargs)
    dim_ld = model.DIM_LD
    flag = (res_estim[-1].last_theta[dim_ld:] != 0).sum() == 0

    return res_estim, flag


def one_result(args):

    prngkey, N, J, P, data, lbd_set, save_all = args
    model = create_cox_mem_jm(pkMEM, N, J, P)

    list_sdg_results, bic = regularization_path(
        estim_fct_with_flag=estim_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        dim_ld=model.DIM_LD,
        N=model.N * (1 + model.J),
        verbatim=True,
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


if __name__ == "__main__":
    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=15)
    myModel = create_cox_mem_jm(pkMEM, N=100, J=15, P=5)

    p_star = myModel.new_params(
        mu1=8,
        mu2=6,
        mu3=40,
        gamma2_1=0.2,
        gamma2_2=0.1,
        sigma2=1e-3,
        alpha=110.1,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, _ = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

    # multi_estim = MultiRunRes(
    #     [
    #         one_estim(jrd.PRNGKey(key), myModel, myobs, lbd=0.1, save_all=True)
    #         for key in range(2)
    #     ]
    # )
    res = one_result((jrd.PRNGKey(0), 100, 15, 5, myobs, my_lbd_set, True))

    # print(multi_estim.chrono)

    # # === PLOT === #

    # names = myModel.params_names
    # sdgplt.plot(multi_estim, dim_ld=7, params_star=p_star, params_names=names)
    # print(f"chrono = {res.chrono}")

    # === PLOT === #
    params_names = myModel.params_names

    sdgplt.plot_theta(res, 7, p_star, params_names)
    sdgplt.plot_theta_hd(res, 7, p_star, params_names)
    sdgplt.plot_reg_path(res, myModel.DIM_LD)
    print(f"chrono = {res.chrono}")
