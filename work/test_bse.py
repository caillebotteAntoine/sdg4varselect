# Create by antoine.caillebotte@inrae.fr
from time import time
import numpy as np
from sdg4varselect import jrd
from sdg4varselect.logistic_model import (
    logistic_curve_float,
)

from sdg4varselect.data_generation import data_simulation


# ===================================#
from collections import namedtuple

params_weibull = namedtuple(
    "params_weibull",
    ("mu1", "mu2", "mu3", "gamma2_1", "gamma2_2", "sigma2", "a", "b", "alpha", "beta"),
)

beta = np.zeros(shape=(100,))
beta[0] = -2
beta[1] = -1
beta[2] = 1
beta[3] = 2

params_star_weibull = params_weibull(
    mu1=0.3,
    mu2=90.0,
    mu3=7.5,
    gamma2_1=0.0025,
    gamma2_2=20,
    sigma2=0.001,
    a=80.0,
    b=35,
    alpha=11.11,
    beta=beta,
)


def get_sample(key, params_star_weibull, N_IND, DIM_COV, cov_law="uniform"):
    # ====== DATA GENERATION ====== #
    return data_simulation(
        params=params_star_weibull,
        key=key,
        N_IND=N_IND,
        J=20,
        p=DIM_COV,
        t_min=60,
        t_max=135,
        cov_law=cov_law,  # cov_law="normal",  # between -1 and 1
    )


# ===================================#

DIM_COV = 100
N_IND = 100


def balance_survival_exp(cov_law):
    data_set, sim, key = get_sample(
        jrd.PRNGKey(0), params_star_weibull, N_IND, DIM_COV, cov_law
    )

    def bse(t, phi1, phi2, params, cov):
        print(f"cov max = {cov.max()} min = {cov.min()}")

        beta_prod_cov = cov @ params.beta
        logistic = logistic_curve_float(t, phi1, phi2, params.mu3)
        bse = beta_prod_cov + params.alpha * logistic

        print(f"beta^T * cov = {beta_prod_cov}")
        print(f"max = {beta_prod_cov.max()} min  = {beta_prod_cov.min()}")

        print(f"logistic = {params.alpha * logistic}")
        print(
            f"max = {(params.alpha * logistic).max()} min  = {(params.alpha * logistic).min()}"
        )
        print(f"bse = {bse}")
        return np.array(bse)

    return bse(80, sim["phi1"], sim["phi2"], params_star_weibull, data_set["cov"])


print(balance_survival_exp("uniform").mean())
print(balance_survival_exp("normal").mean())
