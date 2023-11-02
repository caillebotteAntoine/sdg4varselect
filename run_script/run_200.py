import numpy as np

sim_parameter = {"DIM_COV": 200, "N_IND": 100, "J_OBS": 5, "CENSORING": 0.2}


DIM_COV = 200
N_IND = 100
J_OBS = 5
CENSORING = 0.0

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
