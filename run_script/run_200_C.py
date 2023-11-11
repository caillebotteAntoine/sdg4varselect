import numpy as np
from sdg4varselect import jnp
from multi_run import multi_run

# ====================================================== #
DIM_COV = 200
N_IND = 100
J_OBS = 5
CENSORING = 0.4

params0 = {
    "mu1": 0.5,  # 1
    "mu2": 50.0,  # 2
    "mu3": 3.0,  # 3
    "gamma2_1": 0.00025,  # 4
    "gamma2_2": 2.0,  # 5
    "sigma2": 0.0001,  # 6
    "alpha": 5.0,  # 7
    "beta": np.random.uniform(-1, 1, size=DIM_COV),
}
lbd_set = 10 ** jnp.linspace(-2, 0, num=15)
# lbd_set = [0.19]

multi_run(params0, lbd_set, N_IND, DIM_COV, J_OBS, CENSORING, nrun=50)
