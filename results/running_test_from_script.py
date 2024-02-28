"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp


from sdg4varselect.models.wcox_mem_joint_model import (
    get_params_star,
    create_logistic_weibull_jm,
)
from results.logistic_model.multi_results import multi_run


import sys

seed = int(sys.argv[1])
print(seed)

lbd_set = 10 ** jnp.linspace(-2, 0, num=15)

def test(N, J, P, nrun=1, censoring=2000):
    my_model = create_logistic_weibull_jm(N, J, P)
    p_star = get_params_star(my_model)

    res, censoring_rate = multi_run(
        jrd.PRNGKey(seed),
        lbd_set,
        p_star,
        my_model,
        nrun=nrun,
        censoring=censoring,
        save_all=False,
    )

    C = "NA" if jnp.isnan(censoring_rate) else int(censoring_rate)
    res.save(my_model, root="files_unmerged", filename_add_on=f"C{C}_S{seed}")

#test(100, 5, 10)
#test(100, 5, 50)

test(100, 5, 100)
#test(200, 5, 100)
#test(300, 5, 100)

#test(100, 5, 30)
#test(100, 5, 100)
#test(100, 5, 200)

#test(100, 5, 500)
#test(100, 5, 800)
