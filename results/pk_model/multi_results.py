"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
from datetime import datetime
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect import sample_model
from sdg4varselect.exceptions import sdg4vsNanError

from sdg4varselect.miscellaneous import step_message
from results.logistic_model.one_result import one_result
from collections import namedtuple

multi_estim_res = namedtuple("multi_estim_res", ("estim_res", "censoring_rate"))


# ====================================================== #
def multi_run(prngkey, lbd_set, params_star, model, nrun, censoring, save_all=True):
    chrono_start = datetime.now()
    print(f'start at {chrono_start.strftime("%d/%m/%Y %H:%M:%S")}')

    prngkey_list = jrd.split(prngkey, num=nrun)

    R = []
    for k in range(nrun):
        print(step_message(k, nrun), end="\r")
        dh = sample_model(
            prngkey_list[k], params_star, model, weibull_censoring_loc=censoring
        )

        args = (
            prngkey_list[k],
            model.N,
            model.J,
            model.DIM_HD,
            dh,
            lbd_set,
            save_all,
        )
        try:
            R.append(
                multi_estim_res(
                    estim_res=one_result(args),
                    censoring_rate=1 - dh.data["delta"].mean(),
                )
            )
        except sdg4vsNanError as err:
            print(f"{err} :  estimation cancelled !")

    chrono_time = datetime.now() - chrono_start
    print(f"duration time = {str(chrono_time)}")

    return R, chrono_time


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.logistic_joint_model import (
        Logistic_JM,
        get_params_star,
    )

    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=5)

    myModel = Logistic_JM(N=100, J=5, DIM_HD=200)
    my_params_star = get_params_star(myModel.DIM_HD)

    res, chrono = multi_run(
        jrd.PRNGKey(0), my_lbd_set, my_params_star, myModel, nrun=1, censoring=2000
    )
    print(chrono)

    # === PLOT === #
    sres = res[0].estim_res

    # sdgplt.plot_theta(
    #     sres.listSDGResults[-1], myModel.DIM_LD, my_params_star, myModel.params_names
    # )
    sdgplt.plot_theta_HD(
        sres.listSDGResults[sres.argmin_bic][-1],
        myModel.DIM_LD,
        my_params_star,
        myModel.params_names,
    )

    sdgplt.plot_reg_path(
        my_lbd_set,
        [res[-1] for res in sres.listSDGResults],
        sres.bic[-1],
        myModel.DIM_LD,
    )
