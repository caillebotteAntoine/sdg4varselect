from datetime import datetime
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.logistic import Logistic_JM, sample_one
from sdg4varselect.algo import NanError

from sdg4varselect.miscellaneous import step_message
from results.logistic_model.one_selection_and_estimation import one_estim_with_selection

# import multiprocessing as mpc
from collections import namedtuple

multi_estim_res = namedtuple("multi_estim_res", ("estim_res", "censoring_rate"))


# ====================================================== #
def multi_estim_with_selection(PRNGKey, lbd_set, model, nrun, CENSORING, save_all=True):
    if not isinstance(CENSORING, (list, tuple)):
        CENSORING = [CENSORING]

    print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    PRNGKey_list = jrd.split(PRNGKey, num=nrun)

    R = []
    for censoring in CENSORING:
        R.append([])
        for k in range(nrun):
            print(step_message(k, nrun), end=f"{censoring}/{CENSORING}\r")
            dh = sample_one(PRNGKey_list[k], model, weibull_censoring_loc=censoring)

            args = (
                PRNGKey_list[k],
                model.N,
                model.J,
                model.DIM_HD,
                dh,
                lbd_set,
                save_all,
            )
            try:
                R[-1].append(
                    multi_estim_res(
                        estim_res=one_estim_with_selection(args),
                        censoring_rate=1 - dh.data["delta"].mean(),
                    )
                )
            except NanError as err:
                print(f"{err} :  estimation cancelled !")

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
    plot_box_plot_HD(jnp.array([r.theta for r in res[0]]), model.DIM_LD, params_star)
