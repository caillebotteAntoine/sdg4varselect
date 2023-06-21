# Create by antoine.caillebotte@inrae.fr

from scipy.optimize import brenth

from sdg4varselect import jnp, jrd
from sdg4varselect.logistic_model import (
    logistic_curve,
    logistic_curve_float,
)


def nlmem_simulation(params, key, N_IND, J, t_min, t_max, *args, **kwargs):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    key1, key2, key3, key_out = jrd.split(key, num=4)

    eps = jnp.sqrt(params.sigma2) * jrd.normal(key1, shape=(N_IND, J))
    time = jnp.linspace(t_min, t_max, num=J)

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
    params, key, N_IND, J, t_min, t_max, cov_law="uniform", *args, **kwargs
):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    obs, sim, key = nlmem_simulation(
        params, key, N_IND, J, t_min, t_max, *args, **kwargs
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
        cov = jrd.normal(key1, shape=(N_IND, DIM_COV))
    if cov_law == "bernoulli":
        cov = jnp.array(jrd.bernoulli(key1, shape=(N_IND, DIM_COV)), dtype=jnp.float32)
    if cov_law == "uniform":
        cov = jnp.array(
            jrd.uniform(key1, minval=-1, maxval=1, shape=(N_IND, DIM_COV)),
            dtype=jnp.float32,
        )

    beta_prod_cov_obs = cov @ params.beta

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

    T = jnp.array(tmp)

    obs.update({"T": T, "cov": cov})
    return obs, sim, key_out
