# Create by antoine.caillebotte@inrae.fr

from scipy.optimize import brenth
import numpy as np
from warnings import warn
from collections import namedtuple

from sdg4varselect import jnp, jrd
from sdg4varselect.linear_model import (
    linear_curve,
    linear_curve_float,
)

# ============================================================= #
# ====================== PARAMIETRIZATION ===================== #
# ============================================================= #
params_type = namedtuple(
    "params_type",
    (
        "mu1",
        "mu2",
        "gamma2_1",
        "gamma2_2",
        "sigma2",
        "alpha",
        "beta_surv",
        "beta_long",
    ),
)

# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #


def lmem_simulation(params, key, N_IND, J_OBS, *args, **kwargs):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    key1, key2, key3, key4, key_out = jrd.split(key, num=5)

    DIM_COV_LONG = params.beta_long.shape[0]

    cov_long = jrd.uniform(key4, minval=-0.1, maxval=0.1, shape=(N_IND, DIM_COV_LONG))
    cov_long = cov_long - cov_long.mean(axis=0)[None, :]
    cov_long = jnp.array(
        cov_long,
        dtype=jnp.float32,
    )
    beta_prod_cov_obs = cov_long @ params.beta_long
    print(beta_prod_cov_obs)

    eps = jnp.sqrt(params.sigma2) * jrd.normal(key1, shape=(N_IND, J_OBS))

    # mimicking yearly appointments
    time = jnp.arange(J_OBS) * 365
    time = jnp.concatenate(jnp.array([[time]] * N_IND), axis=0)
    time += jrd.choice(jrd.PRNGKey(0), jnp.arange(1, 365), time.shape)
    time /= time.shape[1] * 365
    time = time

    phi1 = params.mu1 + jnp.sqrt(params.gamma2_1) * jrd.normal(key2, shape=(N_IND,))
    phi2 = params.mu2 + jnp.sqrt(params.gamma2_2) * jrd.normal(key3, shape=(N_IND,))

    Y = (
        linear_curve(
            time=time,
            intercept=phi1 + beta_prod_cov_obs,
            slope=phi2,
        )
        + eps
    )

    return (
        {
            "Y": Y,
            "time": time,
            "cov_long": cov_long,
        },  # obs
        {"phi1": phi1, "phi2": phi2, "eps": eps},  # sim
        key_out,
    )


def data_simulation(
    params,
    key,
    N_IND,
    J_OBS,
    *args,
    **kwargs,
):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    obs, sim, key = lmem_simulation(params, key, N_IND, J_OBS, *args, **kwargs)

    key1, key2, key_out = jrd.split(key, num=3)

    def f(t, phi1, phi2, beta_prod_cov_surv, beta_prod_cov_long, uni):
        """
        For data generation we seek t such that : P(T <= t ) = U([0,1])         # = 1 - S(t)
                                            ie : S(t) = 1 - U([0,1])

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = 1

        so we seek t such that : - int_0^t lbd(s) ds = log(1 - U([0,1]))
                            ie : int_0^t lbd(s) ds + log(1 - U([0,1])) = 0
        """

        def lbd(t):
            return jnp.exp(
                beta_prod_cov_surv
                + params.alpha
                * linear_curve_float(t, intercept=phi1 + beta_prod_cov_long, slope=phi2)
            )

        t_linspace = jnp.linspace(0, t, num=100)
        return jnp.trapz(y=lbd(t_linspace), x=t_linspace) + jnp.log(1 - uni)

    f = jnp.vectorize(f)

    DIM_COV_SURV = params.beta_surv.shape[0]

    cov_surv = jrd.uniform(key1, minval=-0.1, maxval=0.1, shape=(N_IND, DIM_COV_SURV))
    cov_surv = cov_surv - cov_surv.mean(axis=0)[None, :]
    cov_surv = jnp.array(
        cov_surv,
        dtype=jnp.float32,
    )

    beta_prod_cov_surv = cov_surv @ params.beta_surv
    beta_prod_cov_long = obs["cov_long"] @ params.beta_long

    tmp = [0 for i in range(N_IND)]
    uni = jrd.uniform(key2, shape=(N_IND,))

    for i in range(N_IND):
        # Find a root of a function in the interval [a,b]
        args = (
            sim["phi1"][i],
            sim["phi2"][i],
            beta_prod_cov_surv[i],
            beta_prod_cov_long[i],
            uni[i],
        )
        tmp[i] = brenth(f, a=0, b=100, args=args)

    Tstar = jnp.array(tmp)

    delta = Tstar <= obs["time"].max()  # axis=1)  # jnp.ones(Tstar.shape)  #
    T = jnp.minimum(Tstar, obs["time"].max())  # axis=1))

    # id = obs["time"] <= T[:, None]
    # obs["time"] = jnp.where(id, obs["time"], jnp.nan)

    obs.update({"T": T, "delta": delta, "cov_surv": cov_surv})
    sim.update({"T uncensored": Tstar})
    return obs, sim, key_out


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    DIM_COV = 10
    N_IND = 500

    params_star = params_type(
        mu1=1,
        mu2=1.5,
        gamma2_1=2**2,
        gamma2_2=0.3**2,
        sigma2=0.1**2,
        alpha=0.1,
        beta_surv=10
        * jnp.concatenate([jnp.array([1, 2, 1, 2]), jnp.zeros(shape=(DIM_COV - 4,))]),
        beta_long=jnp.concatenate(
            [jnp.array([0.3, 0.5, 0.3, 0.5]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    obs, sim, _ = data_simulation(params_star, jrd.PRNGKey(0), N_IND, 50)
    print(obs)

    plt.plot(obs["time"].T, obs["Y"].T, "o-")

    plt.figure()
    plt.hist(obs["T"], bins=20)

    print(f'censoring = {int((1-obs["delta"].mean())*100)}%')
