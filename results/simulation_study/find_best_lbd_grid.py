import jax.numpy as jnp
import jax.random as jrd

import sdg4varselect.plot as sdgplt
from sdg4varselect.outputs import MultiRunRes

from NLMEM.Logistic.model import myModel, one_result, one_estim_with_flag, mydata

mylbd_set = 10 ** jnp.linspace(-3, -0, num=10)


def test_lbd_set(lbd_set, num=10):

    reg_res = one_result(
        one_estim_with_flag, jrd.PRNGKey(0), myModel, mydata, lbd_set, save_all=False
    )
    a = lbd_set[reg_res.argmin_bic - 1]
    b = lbd_set[reg_res.argmin_bic + 1]

    return 10 ** jnp.linspace(jnp.log10(a), jnp.log10(b), num=num), reg_res


mylbd_set = 10 ** jnp.linspace(-3, 0, num=10)
estim_res = []
for i in range(2):
    mylbd_set, reg = test_lbd_set(mylbd_set)
    estim_res.append(reg)

    MultiRunRes(estim_res).save(myModel, filename_add_on="find_best_lbd")

estim_res = MultiRunRes.load(myModel, filename_add_on="find_best_lbd")


for r in estim_res:
    fig = sdgplt.plot_reg_path(reg_res=r, dim_ld=myModel.DIM_LD)
