"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd

from results.logistic_model.one_estim import one_estim

from sdg4varselect._estimation_method import lasso_into_adaptive_into_estim
from sdg4varselect.models.wcox_mem_joint_model import (
    create_logistic_weibull_jm,
    get_params_star,
)

if __name__ == "__main__":

    myModel = create_logistic_weibull_jm(100, 5, 10)
    my_params_star = get_params_star(myModel)

    myobs, _ = myModel.sample(my_params_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

    multi_estim = lasso_into_adaptive_into_estim(
        one_estim, jrd.PRNGKey(0), myModel, myobs, 0.5 * 10**-1, save_all=True
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
