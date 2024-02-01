"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
from datetime import datetime
import jax.random as jrd
import jax.numpy as jnp

import sdg4varselect.models.logistic_joint_model as modelisation
from sdg4varselect.exceptions import sdg4vsNanError

from sdg4varselect.miscellaneous import step_message
from results.logistic_model.one_result import one_result
from collections import namedtuple

multi_estim_res = namedtuple("multi_estim_res", ("estim_res", "censoring_rate"))


# ====================================================== #
def multi_estim_with_selection(PRNGKey, lbd_set, model, nrun, CENSORING, save_all=True):
    if not isinstance(CENSORING, (list, tuple)):
        CENSORING = [CENSORING]

    chrono_start = datetime.now()
    print(f'start at {chrono_start.strftime("%d/%m/%Y %H:%M:%S")}')
    PRNGKey_list = jrd.split(PRNGKey, num=nrun)

    R = []
    for censoring in CENSORING:
        R.append([])
        for k in range(nrun):
            print(step_message(k, nrun), end=f"{censoring}/{CENSORING}\r")
            dh = modelisation.sample_one(
                PRNGKey_list[k], model, weibull_censoring_loc=censoring
            )

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
                        estim_res=one_result(args),
                        censoring_rate=1 - dh.data["delta"].mean(),
                    )
                )
            except sdg4vsNanError as err:
                print(f"{err} :  estimation cancelled !")

    chrono_time = datetime.now() - chrono_start
    print(f"end at {str(chrono_time)}")

    return R, chrono_time


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt

    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=5)

    myModel = modelisation.Logistic_JM(N=50, J=5, DIM_HD=10)

    res = multi_estim_with_selection(
        jrd.PRNGKey(0), my_lbd_set, myModel, nrun=1, CENSORING=2000
    )

    # === PLOT === #
    params_star = modelisation.get_params_star(myModel.DIM_HD)
    sres, chrono = res[0][0][0]
    print(chrono)

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
