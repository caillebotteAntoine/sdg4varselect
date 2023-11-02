import numpy as np
from datetime import datetime
import pickle
from joint_model.one_res import method
from joint_model.sample import get_params_star

from joint_model.one_run import get_random_params0

from time import time
import jax.random as jrd
import jax.numpy as jnp

# ====================================================== #
print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

seed = 1697807204  #
seed = int(time())  #
print(f"seed = {seed}")
prng_key = jrd.PRNGKey(seed)
# ====================================================== #

DIM_COV = 200
N_IND = 100
J_OBS = 5
CENSORING = 0

params_star_stack, params_star_weibull = get_params_star(DIM_COV)

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
lbd_set = 10 ** jnp.linspace(-2, 0, num=10)
# lbd_set = [0.19]
# ====================================================== #

lrf = []
lrs = []
llbd = []
lbic = []
lebic = []
ltheta_reg = []

print(f"n = {N_IND}, p = {DIM_COV}, J = {J_OBS}")
print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

nrun = 50
for i in range(nrun):
    # print(step_message(i, nrun))
    params_start, prng_key = get_random_params0(prng_key, params0, error=0.2)

    res = method(
        params0,
        params_star_weibull,
        lbd_set,
        N_IND,
        DIM_COV,
        J_OBS,
        CENSORING,
        prng_key=prng_key,
        verbatim=True,
    )
    if res != -1:
        res_f, _, solver, res_s, bic, ebic, theta_reg, lbd_select, _, _ = res
        lrf.append(res_f)
        lrs.append(res_s)
        llbd.append(lbd_select)
        lbic.append(bic)
        lebic.append(ebic)
        ltheta_reg.append(theta_reg)

print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')


theta = [res.theta[-1] for res in lrf]
theta_biais = [res.theta[-1] for res in lrs]

data = {
    "theta": theta,
    "theta_biais": theta_biais,
    "lbd_select": llbd,
    # "params_star_weibull": params_star_weibull,
    "params_star_stack": params_star_stack,
    "params_names": solver.params_names,
    "lbic": lbic,
    "lebic": lebic,
    "ltheta_reg": ltheta_reg,
    "lbd_set": lbd_set,
}

with open("res_multi_run.pkl", "wb") as f:
    pickle.dump(data, f)

print("RESULT SAVED !")
