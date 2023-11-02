# Create by antoine.caillebotte@inrae.fr

from scipy.optimize import brenth
import numpy as np
from warnings import warn

from sdg4varselect import jnp, jrd
from sdg4varselect.logistic_model import (
    logistic_curve,
    logistic_curve_float,
)


def nlmem_simulation(params, key, N_IND, J_OBS, t_min, t_max, *args, **kwargs):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    key1, key2, key3, key_out = jrd.split(key, num=4)

    eps = jnp.sqrt(params.sigma2) * jrd.normal(key1, shape=(N_IND, J_OBS))
    time = jnp.linspace(t_min, t_max, num=J_OBS)

    phi1 = params.mu1 + jnp.sqrt(params.gamma2_1) * jrd.normal(key2, shape=(N_IND,))
    phi2 = params.mu2 + jnp.sqrt(params.gamma2_2) * jrd.normal(key3, shape=(N_IND,))
    phi3 = jnp.array([params.mu3])

    Y = (
        logistic_curve(
            time=time,
            supremum=phi1,
            midpoint=phi2,
            growth_rate=phi3,
        )
        + eps
    )

    return (
        {
            "Y": Y,
            "time": time,
        },  # obs
        {"phi1": phi1, "phi2": phi2},  # sim
        key_out,
    )


def data_simulation(
    params,
    key,
    N_IND,
    J_OBS,
    t_min,
    t_max,
    cov_law="uniform",
    censoring=0.0,
    *args,
    **kwargs,
):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    obs, sim, key = nlmem_simulation(
        params, key, N_IND, J_OBS, t_min, t_max, *args, **kwargs
    )

    key1, key2, key_out = jrd.split(key, num=3)

    def f(t, phi1, phi2, beta_prod_cov, uni):
        """
        For data generation we seek t such that : P(T <= t ) = U([0,1])         # = 1 - S(t)
                                            ie : S(t) = 1 - U([0,1])

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

        so we seek t such that : - int_0^t lbd(s) ds = log(1 - U([0,1]))
                            ie : int_0^t lbd(s) ds + log(1 - U([0,1])) = 0
        """

        def lbd(t):
            lbd0 = params.b / params.a * (t / params.a) ** (params.b - 1)
            return lbd0 * jnp.exp(
                beta_prod_cov
                + params.alpha * logistic_curve_float(t, phi1, phi2, params.mu3)
            )

        t_linspace = jnp.linspace(0, t, num=100)
        return jnp.trapz(y=lbd(t_linspace), x=t_linspace) + jnp.log(1 - uni)

    f = jnp.vectorize(f)

    DIM_COV = params.beta.shape[0]

    if cov_law == "normal":
        cov = jrd.normal(key1, shape=(N_IND, DIM_COV))  # N(0,1)
        cov = cov / (cov.mean(axis=0)[None, :])
        cov = cov / (cov.var(axis=0)[None, :] ** 0.5)
        # print(f"mean = {cov.mean(axis = 0)} var = {cov.var(axis = 0)}")

    elif cov_law == "bernoulli":
        cov = jrd.bernoulli(key1, shape=(N_IND, DIM_COV))
        cov = jnp.array(
            cov / max(cov.max(), -cov.min()),
            dtype=jnp.float32,
        )
    elif cov_law == "uniform":
        cov = jrd.uniform(key1, minval=-1, maxval=1, shape=(N_IND, DIM_COV))

        cov = cov - cov.mean(axis=0)[None, :]

    elif cov_law == "clever_uniform":
        k1, k2, k3, k4, k5 = jrd.split(key1, num=5)

        cov1 = jrd.uniform(k1, minval=0.4, maxval=0.6, shape=(N_IND,))
        cov2 = jrd.uniform(k2, minval=-0.6, maxval=-0.4, shape=(N_IND,))
        cov3 = jrd.uniform(k3, minval=0.4, maxval=0.6, shape=(N_IND,))
        cov4 = jrd.uniform(k4, minval=-0.6, maxval=-0.4, shape=(N_IND,))

        cov5 = jrd.uniform(k5, minval=-0.5, maxval=0.5, shape=(DIM_COV - 4, N_IND))

        cov = jnp.row_stack((cov1, cov2, cov3, cov4, cov5)).T

    elif cov_law == "NIRS":
        import pandas as pd

        dt = pd.read_csv("../simulated_nirs.csv", sep=";", header=1, decimal=",")
        cov = np.array(dt.values)[:, 0:N_IND]

        p_max = cov.shape[0]
        p_step = int(np.round(p_max / DIM_COV))

        cov = cov[[i * p_step for i in range(DIM_COV)]].T
        cov = cov - cov.mean(axis=0)[None, :]

    cov = jnp.array(
        cov,
        dtype=jnp.float32,
    )

    beta_prod_cov_obs = cov @ params.beta

    # print(f"beta^T cov = {beta_prod_cov_obs}")

    tmp = [0 for i in range(N_IND)]
    uni = jrd.uniform(key2, shape=(N_IND,))

    for i in range(N_IND):
        # Find a root of a function in the interval [a,b]
        args = (
            sim["phi1"][i],
            sim["phi2"][i],
            beta_prod_cov_obs[i],
            uni[i],
        )
        tmp[i] = brenth(f, a=0, b=10 * params.a, args=args)

    Tstar = jnp.array(tmp)

    C = (
        Tstar.sort()[int(N_IND * (1 - censoring))]
        if censoring != 0.0
        else Tstar.max() * 2
    )
    T = np.array([min(Tstar[i], C) for i in range(N_IND)])
    delta = Tstar <= C

    if (obs["time"] < C).sum() < J_OBS:
        J_NEW = int((obs["time"] < C).sum())
        warn(
            f"censuring implies to shrink the longitudinal data to {J_NEW} observations ! \n censuring starting at {C}"
        )

    id = [i for i in range(J_OBS) if obs["time"][i] < C]
    obs["Y"] = obs["Y"][:, id]
    obs["time"] = obs["time"][obs["time"] < C]

    obs.update({"T": T, "delta": delta, "cov": cov})
    sim.update({"T uncensored": Tstar})
    return obs, sim, key_out
