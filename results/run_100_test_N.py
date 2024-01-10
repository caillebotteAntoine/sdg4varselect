import jax.random as jrd
import jax.numpy as jnp

import gzip
import pickle


from sdg4varselect.logistic import Logistic_JM
from logistic_model.multi_estim import multi_estim_with_selection

lbd_set = 10 ** jnp.linspace(-2, 0, num=15)


def testN(N):
    model = Logistic_JM(N=N, J=5, DIM_HD=20)

    seed = 0
    R = multi_estim_with_selection(
        jrd.PRNGKey(seed), lbd_set, model, nrun=2, CENSORING=2000
    )

    res = {"res": R, "lbd_set": lbd_set, "N": model.N, "J": model.J, "P": model.DIM_HD}

    filename = f"s{seed}_N{model.N}_P{model.DIM_HD}_J{model.J}"
    # _C{int(jnp.array(lcensoring_rate).mean()*100)}"
    pickle.dump(res, gzip.open(f"files/testN_{filename}.pkl.gz", "wb"))
    print(f"{filename} SAVED !")


testN(5)
testN(10)
testN(20)
testN(30)
