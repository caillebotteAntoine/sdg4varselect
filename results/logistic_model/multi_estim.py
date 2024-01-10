from datetime import datetime
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.logistic import Logistic_JM, sample_one

from results.logistic_model.one_selection_and_estimation import one_estim_with_selection


# ====================================================== #
def multi_estim_with_selection(PRNGKey, lbd_set, model, nrun, CENSORING, save_all):
    if not isinstance(CENSORING, list):
        CENSORING = [CENSORING]

    print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    R = []
    for censoring in CENSORING:
        dh = sample_one(PRNGKey, model, weibull_censoring_loc=censoring)
        args = (
            (
                jrd.PRNGKey(i),
                model.N,
                model.J,
                model.DIM_HD,
                dh,
                lbd_set,
                save_all,
            )
            for i in range(nrun)
        )

        R.append([one_estim_with_selection(k) for k in args])

    print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    return R


if __name__ == "__main__":
    lbd_set = 10 ** jnp.linspace(-2, 0, num=5)

    model = Logistic_JM(N=100, J=5, DIM_HD=10)

    res = multi_estim_with_selection(
        jrd.PRNGKey(0), lbd_set, model, nrun=2, CENSORING=2000
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

    from sdg4varselect.logistic import get_params_star

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
