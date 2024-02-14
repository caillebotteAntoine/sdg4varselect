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

# from sdg4varselect.outputs import MultiRunRes

nrun = 20
lbd_set = 10 ** jnp.linspace(-2, 0, num=15)


# def testC(*censoring_loc):
#     model = Logistic_JM(N=100, J=5, DIM_HD=200)

#     seed = 1
#     R = multi_run(
#         jrd.PRNGKey(seed),
#         lbd_set,
#         model,
#         nrun=nrun,
#         CENSORING=censoring_loc,
#         save_all=False,
#     )

#     res = {"res": R, "lbd_set": lbd_set, "N": model.N, "J": model.J, "P": model.DIM_HD}

#     filename = f"s{seed}_N{model.N}_P{model.DIM_HD}_J{model.J}"
#     # _C{int(jnp.array(lcensoring_rate).mean()*100)}"
#     pickle.dump(res, gzip.open(f"files/test_{filename}.pkl.gz", "wb"))
#     print(f"{filename} SAVED !")


def test(N, J, P, censoring=2000):
    my_model = create_logistic_weibull_jm(N, J, P)
    p_star = get_params_star(my_model)

    seed = 2
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
    res.save(my_model, root="files", filename_add_on=f"C{C}_S{seed}")


# for i in (50,):
#     testN(i)
# import jax
# import os


# print(jax.device_count(backend="cpu"))

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=<1>"
# import os

# os.environ["XLA_FLAGS"] = "--xla_cpu_force_host_platform_device_count=4"

# print(jax.device_count(backend="cpu"))

# testC(1000, 85, 77)

test(100, 5, 10)
test(100, 5, 50)

test(100, 5, 100)
test(200, 5, 100)
test(300, 5, 100)

test(100, 5, 30)
test(100, 5, 100)
test(100, 5, 200)


test(100, 5, 500)
test(100, 5, 800)


# for i in (10, 50, 200, 400, 600, 1000):
#     testP(i)
