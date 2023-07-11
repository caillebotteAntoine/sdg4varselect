# Create by antoine.caillebotte@inrae.fr
from time import time
from sdg4varselect.miscellaneous import time2string

from sdg4varselect.solver import shrink_support
from random import uniform
import numpy as np
import pandas

from one_run import (
    kwargs_run_GD,
    sample,
    estim,
    estim_solver,
    sample_and_estim,
    params0,
    N_IND,
    DIM_COV,
    params_star_stack,
)
from sdg4varselect.miscellaneous import step_message

from sdg4varselect import jrd, jnp
import sdg4varselect.plot as sdgplt

lbd_selection = 0.29
folder = "29"


def clever_regularization_path(path, prng_key, nrep=1, verbatim=False):
    res_solver = []
    data_set, key = sample(params0, prng_key)

    for i in range(len(path)):
        print(step_message(i, len(path)), end="\r" if not verbatim else "\n")

        kwargs_run_GD["prox_regul"] = path[i]
        kwargs_run_GD["proximal_operator"] = True
        _, solver, key = estim(data_set, key, verbatim=verbatim)
        res_solver.append(solver)

        print(f"#beta = {solver.get_number_of_nonzero(p=DIM_COV)}")

        if solver.get_number_of_nonzero(p=DIM_COV) == 0:
            for k in range(len(path) - i - 1):
                print(
                    step_message(i + k, len(path)), end="\r" if not verbatim else "\n"
                )
                res_solver.append(res_solver[-1])
            print(f"break at {path[i]}")
            break

    return res_solver, key


# ====================================================== #
# ================ REGULARIZATION PATH ================= #
# ====================================================== #
lbd_set = 10 ** jnp.linspace(-2, 0, num=50)

res_solver, prng_key = clever_regularization_path(
    lbd_set, jrd.PRNGKey(0), verbatim=False
)


_, _, bic = sdgplt.plot_regularization_path(res_solver, lbd_set, p=DIM_COV, N=N_IND)
