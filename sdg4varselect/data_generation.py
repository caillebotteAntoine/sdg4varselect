# Create by antoine.caillebotte@inrae.fr

from scipy.optimize import brenth
import numpy as np
from warnings import warn

from sdg4varselect import jnp, jrd


# ======== TUTO FCT PYTHON PYTEST======== #
def f(a, b, *args, **kwargs):
    """args is list, kwargs is a dict"""
    if len(args) != 0 and len(kwargs) == 0:
        return a * b + args[0]
    if "d" in kwargs:
        return a * b + args[0] * kwargs["d"]

    return a * b


def test_f():
    assert f(2, 3) == 6
    # etc


# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #
def cov_simulation(PRNGKey, min, max, shape):
    key, PRNGKey = jrd.split(PRNGKey, num=2)

    cov = jrd.uniform(key, minval=min, maxval=max, shape=shape)
    cov = cov - cov.mean(axis=0)[None, :]
    cov = jnp.array(cov, dtype=jnp.float32)

    return cov, PRNGKey


def mem_simulation(
    params,
    PRNGKey,
    N_IND,
    noise_variance,  # = sigma2
    fct,  # = logistic_curve
    random_effects,  # = { "phi1":("mu1", "gamma2_1"), "phi2" }
    fixed_effets,  # = { "phi3":("mu3")}
    fct_kwargs,  # other parameters, example = [time]
):
    """return simulation following mixed effect model
    Y = fct(random_effects, fixed_effects,kwargs) + N(0, noise_variance^2)
    """

    sim = {}
    for name, value in random_effects.items():
        key, PRNGKey = jrd.split(PRNGKey, num=2)

        mean = getattr(params, value[0])
        var = getattr(params, value[1])
        # N(mean, var^2)
        sim[name] = mean + jnp.sqrt(var) * jrd.normal(key, shape=(N_IND,))
        fct_kwargs[name] = sim[name]

    for name, value in fixed_effets.items():
        sim[name] = jnp.array([getattr(params, value)])
        fct_kwargs[name] = sim[name]

    Y_without_noise = fct(**fct_kwargs)

    key, PRNGKey = jrd.split(PRNGKey, num=2)
    sim["eps"] = jnp.sqrt(getattr(params, noise_variance)) * jrd.normal(
        key, shape=Y_without_noise.shape
    )

    Y = Y_without_noise + sim["eps"]

    return (
        {"Y": Y},  # obs
        sim,
        PRNGKey,
    )


def cox_simulation(
    params, PRNGKey, beta_prod_cov, baseline_fct, baseline_kwargs, link_fct, link_kwargs
):
    """
    lbd(t) = baseline_fct(t) * exp(beta^T U + linkfct(alpha, t, ...))
    """

    (N_IND,) = beta_prod_cov.shape

    def f(t, beta_prod_cov_ind, uni, *link_ind_args):
        """
        example if baseline_fct = lbd_0 = weibull and linkfct = m the logistic function

        For data generation we seek t such that : P(T <= t ) = U([0,1])         # = 1 - S(t)
                                            ie : S(t) = 1 - U([0,1])

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

        so we seek t such that : - int_0^t lbd(s) ds = log(1 - U([0,1]))
                            ie : int_0^t lbd(s) ds + log(1 - U([0,1])) = 0      # f(t) = 0
        """

        def lbd(s):
            """baseline_fct * exp[ beta^T U + linkfct(alpha, M(t, ...)) ]"""
            lbd0 = baseline_fct(s, **baseline_kwargs)
            return lbd0 * jnp.exp(
                beta_prod_cov_ind + link_fct(s, params.alpha, *link_ind_args)
            )

        t_linspace = jnp.linspace(0, t, num=100)
        return jnp.trapz(y=lbd(t_linspace), x=t_linspace) + jnp.log(1 - uni)

    f = jnp.vectorize(f)

    tmp = [0 for i in range(N_IND)]
    key, PRNGKey = jrd.split(PRNGKey, num=2)
    # rem : c'est pas pratique si uni est très très petit le log explose ...
    uni = jrd.uniform(key, shape=(N_IND,))

    for i in range(N_IND):
        args = [beta_prod_cov[i], uni[i]]

        # extract individual argument for the link function
        for value in link_kwargs.values():
            if isinstance(value, (np.ndarray, jnp.ndarray)) and value.shape != ():
                args.append(value[i])
            else:
                args.append(value)

        # Find a root of a function in the interval [a,b]
        tmp[i] = brenth(f, a=0, b=1000, args=tuple(args))

    Tstar = jnp.array(tmp)
    sim = {"T uncensored": Tstar}

    return {}, sim, PRNGKey


if __name__ == "__main__":
    pass
