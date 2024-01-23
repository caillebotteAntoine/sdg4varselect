import jax.random as jrd
import jax.numpy as jnp

import gzip
import pickle


from sdg4varselect.logistic import Logistic_JM
from logistic_model.multi_estim import multi_estim_with_selection

nrun = 2
lbd_set = 10 ** jnp.linspace(-2, 0, num=5)


def testC(*censoring_loc):
    model = Logistic_JM(N=100, J=5, DIM_HD=200)

    seed = 1
    R = multi_estim_with_selection(
        jrd.PRNGKey(seed),
        lbd_set,
        model,
        nrun=nrun,
        CENSORING=censoring_loc,
        save_all=False,
    )
    res = {"res": R, "lbd_set": lbd_set, "N": model.N, "J": model.J, "P": model.DIM_HD}

    filename = f"s{seed}_N{model.N}_P{model.DIM_HD}_J{model.J}"
    # _C{int(jnp.array(lcensoring_rate).mean()*100)}"
    pickle.dump(res, gzip.open(f"files/test_{filename}.pkl.gz", "wb"))
    print(f"{filename} SAVED !")


def testN(N):
    model = Logistic_JM(N=N, J=5, DIM_HD=800)

    seed = 1
    R = multi_estim_with_selection(
        jrd.PRNGKey(seed), lbd_set, model, nrun=nrun, CENSORING=2000, save_all=False
    )
    res = {"res": R, "lbd_set": lbd_set, "N": model.N, "J": model.J, "P": model.DIM_HD}

    filename = f"s{seed}_N{model.N}_P{model.DIM_HD}_J{model.J}"
    # _C{int(jnp.array(lcensoring_rate).mean()*100)}"
    pickle.dump(res, gzip.open(f"files/test_{filename}.pkl.gz", "wb"))
    print(f"{filename} SAVED !")


def testP(P):
    model = Logistic_JM(N=100, J=5, DIM_HD=P)

    seed = 1
    R = multi_estim_with_selection(
        jrd.PRNGKey(seed), lbd_set, model, nrun=nrun, CENSORING=2000, save_all=False
    )
    res = {"res": R, "lbd_set": lbd_set, "N": model.N, "J": model.J, "P": model.DIM_HD}

    filename = f"s{seed}_N{model.N}_P{model.DIM_HD}_J{model.J}"
    # _C{int(jnp.array(lcensoring_rate).mean()*100)}"
    pickle.dump(res, gzip.open(f"files/test_{filename}.pkl.gz", "wb"))
    print(f"{filename} SAVED !")


# for i in (50,):
#     testN(i)

testC(1000, 85, 77)

for i in (30, 100, 200, 400, 600):
    testN(i)

for i in (5, 200, 400, 600, 1000):
    testP(i)
