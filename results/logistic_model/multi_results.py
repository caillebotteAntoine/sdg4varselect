"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
from datetime import datetime
import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.outputs import MultiRegRes
import sdg4varselect.plot as sdgplt
from sdg4varselect.miscellaneous import step_message

from sdg4varselect.models import WeibullCoxJM, logisticMEM

from results.logistic_model.one_result import one_result


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
    my_lbd_set = 10 ** jnp.linspace(-2, 0, num=10)
    # my_lbd_set = [1.5 * 10**-1]

    myModel = WeibullCoxJM(
        logisticMEM(N=500, J=15), P=5, alpha_scale=0.001, a=800, b=10
    )

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

    res, C = multi_run(
        jrd.PRNGKey(0), my_lbd_set, p_star, myModel, nrun=50, censoring=2000
    )
    print(C)

    # res = sdg4vsResults.load(myModel)

    # === PLOT === #
    params_names = myModel.params_names

    sdgplt.plot_theta(res[0], 7, p_star, params_names)
    sdgplt.plot_theta_hd(res[0], 7, p_star, params_names)
    sdgplt.plot_reg_path(res[0], myModel.DIM_LD)
    print(f"chrono = {res.chrono}")

    # === PLOT === #

    # sdgplt.plot_theta(
    #     sres.listSDGResults[-1], myModel.DIM_LD, params_star, myModel.params_names
    # )

    # sdgplt.plot(res[0], myModel.DIM_LD)
