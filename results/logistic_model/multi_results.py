"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
from datetime import datetime
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.exceptions import sdg4vsNanError

from sdg4varselect.miscellaneous import step_message
from results.logistic_model.one_result import one_result

from sdg4varselect.outputs import MultiRegRes

from sdg4varselect.outputs import sdg4vsResults


# ====================================================== #
def multi_run(prngkey, lbd_set, params_star, model, nrun, censoring, save_all=True):
    chrono_start = datetime.now()
    print(f'start at {chrono_start.strftime("%d/%m/%Y %H:%M:%S")}')

    prngkey_list = jrd.split(prngkey, num=nrun)

    estim_res = []
    censoring_rate = []
    end = "\r" if __name__ == "__main__" else "\n"
    for k in range(nrun):
        print("run", step_message(k, nrun), end=end)
        data, _ = model.sample(
            params_star, prngkey_list[k], weibull_censoring_loc=censoring
        )

        args = (
            prngkey_list[k],
            model.N,
            model.J,
            model.P,
            data,
            lbd_set,
            save_all,
        )
        try:
            estim_res.append(
                one_result(args),
            )
            censoring_rate.append(1 - data["delta"].mean())

        except sdg4vsNanError as err:
            print(f"{err} :  estimation cancelled !")

    return MultiRegRes(estim_res), jnp.array(censoring_rate).mean()


if __name__ == "__main__":
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.wcox_mem_joint_model import (
        get_params_star,
        create_logistic_weibull_jm,
    )

    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=5)
    myModel = create_logistic_weibull_jm(100, 5, 10)
    p_star = get_params_star(myModel)

    # res, censoring = multi_run(
    #     jrd.PRNGKey(0), my_lbd_set, p_star, myModel, nrun=2, censoring=2000
    # )
    # print(censoring)

    res = sdg4vsResults.load(myModel)

    # === PLOT === #

    # sdgplt.plot_theta(
    #     sres.listSDGResults[-1], myModel.DIM_LD, params_star, myModel.params_names
    # )

    sdgplt.plot(
        res[0].final_result,
        dim_ld=myModel.DIM_LD,
        params_star=p_star,
        params_names=myModel.params_names,
    )

    sdgplt.plot(res[0], myModel.DIM_LD)
