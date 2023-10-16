from one_run import (
    get_random_params0,
    params_star_weibull,
    params_star_stack,
    N_IND,
    DIM_COV,
)

import pickle
import sdg4varselect.plot as sdgplt
from sdg4varselect import jrd, jnp
from time import time
from one_res import clever_regularization_path, final_estim
import numpy as np

from sdg4varselect.miscellaneous import step_message, bic_final_estim_from_list

from sdg4varselect.miscellaneous import time2string

lbd_set = 10 ** jnp.linspace(-2, 0, num=5)


def method(nrun=1, verbatim=True):
    params0, prng_key = get_random_params0(jrd.PRNGKey(int(time())), error=0.2)

    # ====================================================== #
    # ================ REGULARIZATION PATH ================= #
    # ====================================================== #
    ls, lr = [], []
    for k in range(nrun):
        time_start = time()
        s, r, prng_key = clever_regularization_path(
            params0, lbd_set, prng_key, verbatim=verbatim
        )
        print(f"REGULARIZATION PATH TIME: {time2string(time() - time_start)}")

        ls.append(s)
        lr.append(r)

    bic, ebic, theta_reg = bic_final_estim_from_list(
        ls, lr, N_IND, DIM_COV, verbatim=verbatim
    )

    # ============================================ #
    # ================ INFERENCE ================= #
    # ============================================ #
    bic_argmin = np.argmin(bic)
    solver_selection = ls[0][bic_argmin]

    final_res, _ = final_estim(
        solver_selection, params0, lbd_set[bic_argmin], verbatim=True
    )

    return final_res, bic, ebic, theta_reg


lr = []

for i in range(5):
    print(step_message(i, 5))

    res, _, _, _ = method(verbatim=False)
    lr.append(res)

theta = [res.theta[-1] for res in lr]

data = {"theta": theta}

with open("res_multi_run.pkl", "wb") as f:
    pickle.dump(data, f)


res = lr[0]
_, _ = sdgplt.plot_params_grad(
    res.theta,
    res.grad_precond,
    np.array(params_star_stack),
    p=DIM_COV,
    logscale=False,
)

_, _ = sdgplt.plot_params_hd(res.theta, p=DIM_COV, location="right")


# fig, axs = sdgplt.plot_regularization_path(theta_reg, lbd_set, bic)
# ax, ax_bic = axs

# ax_ebic = ax.twinx()
# ax_ebic.plot(lbd_set, ebic, color="r", linewidth=2, linestyle="--", label="eBIC")
# id_min = np.nanargmin(ebic)
# sdgplt.plot_axvline(ax_ebic, lbd_set, ebic, id_min, color="g", msg="min(eBIC)")
# ax_ebic.legend(loc="upper right")


theta = np.array([res.theta[-1,] for res in lr])
