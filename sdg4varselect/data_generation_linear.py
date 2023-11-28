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
params_weibull = namedtuple(
    "params_weibull",
    ("mu1", "mu2", "gamma2_1", "gamma2_2", "sigma2", "a", "b", "alpha", "beta"),
)

# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #


def nlmem_simulation(params, key, N_IND, J_OBS, t_min, t_max):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    key1, key2, key3, key_out = jrd.split(key, num=4)

    eps = jnp.sqrt(params.sigma2) * jrd.normal(key1, shape=(N_IND, J_OBS))
    time = jnp.linspace(t_min, t_max, num=J_OBS)

    phi1 = params.mu1 + jnp.sqrt(params.gamma2_1) * jrd.normal(key2, shape=(N_IND,))
    phi2 = params.mu2 + jnp.sqrt(params.gamma2_2) * jrd.normal(key3, shape=(N_IND,))

    Y = (
        linear_curve(
            time=time,
            intercept=phi1,
            slope=phi2,
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
    censoring=0.0,
):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    obs, sim, key = nlmem_simulation(params, key, N_IND, J_OBS, t_min, t_max)

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
                beta_prod_cov + params.alpha * linear_curve_float(t, phi1, phi2)
            )

        t_linspace = jnp.linspace(0, t, num=100)
        return jnp.trapz(y=lbd(t_linspace), x=t_linspace) + jnp.log(1 - uni)

    f = jnp.vectorize(f)

    DIM_COV = params.beta.shape[0]

    cov = jrd.uniform(key1, minval=-0.1, maxval=0.1, shape=(N_IND, DIM_COV))
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    DIM_COV = 10
    N_IND = 500

    params_star = params_weibull(
        mu1=1,
        mu2=1.5,
        gamma2_1=0.1**2,
        gamma2_2=0.1**2,
        sigma2=0.1**2,
        a=2.0,
        b=5,
        alpha=1.5,
        beta=jnp.concatenate(
            [jnp.array([0.3, 0.5, 0.3, 0.5]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    obs, sim, _ = data_simulation(
        params_star, jrd.PRNGKey(0), N_IND, 50, t_min=0, t_max=1, censoring=0.0
    )
    print(obs)

    plt.plot(obs["time"].T, obs["Y"].T, "o-")

    plt.figure()
    plt.hist(obs["T"], bins=20)

    print(f'censoring = {int((1-obs["delta"].mean())*100)}%')
