from datetime import datetime
import pickle
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.logistic import Logistic_JM, sample_one, get_params_star

from results.logistic_model.one_estim import algo_settings
from results.logistic_model.one_selection_and_estimation import one_estim_with_selection

# import multiprocessing as mpc
# from tqdm import tqdm


# ====================================================== #
def multi_estim_with_selection(PRNGKey, lbd_set, model, nrun, algo_settings, CENSORING):
    if not isinstance(CENSORING, list):
        CENSORING = [CENSORING]

    print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    R = []
    for censoring in CENSORING:
        dh = sample_one(PRNGKey, model, weibull_censoring_loc=censoring)
        args = (
            (jrd.PRNGKey(i), model.N, model.J, model.DIM_HD, dh, lbd_set)
            for i in range(nrun)
        )

        R.append([one_estim_with_selection(k) for k in args])

    print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    return R


if __name__ == "__main__":
    lbd_set = 10 ** jnp.linspace(-2, 0, num=5)

    model = Logistic_JM(N=50, J=5, DIM_HD=5)

    res = multi_estim_with_selection(
        jrd.PRNGKey(0), lbd_set, model, 2, algo_settings, 2000
    )

    # === PLOT === #
    from sdg4varselect.plot import (
        plot_theta,
        plot_reg_path,
        plot_theta_HD,
        plot_box_plot_HD,
    )

    selection_res = res[0][0]
    reg_path = selection_res.regularization_path

    params_star = get_params_star(model.DIM_HD)

    plot_theta(reg_path, model.DIM_LD, params_star, model.params_names)
    plot_reg_path(lbd_set, reg_path, selection_res.bic, model.DIM_HD)

    plot_theta(
        [r.estim_res for r in res[0]], model.DIM_LD, params_star, model.params_names
    )
    plot_theta_HD(
        [r.estim_res for r in res[0]], model.DIM_LD, params_star, model.params_names
    )
    plot_box_plot_HD([r.estim_res for r in res[0]], model.DIM_LD, params_star)
