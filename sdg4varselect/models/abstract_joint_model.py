"""
Module for abstract class AbstractJointModel.

Create by antoine.caillebotte@inrae.fr
"""
# pylint: disable=C0116

from abc import abstractmethod
import functools

import numpy as np
from scipy.optimize import brenth


import jax.numpy as jnp
import jax.random as jrd
from jax import jit, jacfwd

import parametrization_cookbook.jax as pc


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    """Computation of the current target distribution score"""
    out = jnp.log(2 * jnp.pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


class AbstractJointModel:
    def __init__(self, N=1, J=1, DIM_HD=1, **kwargs):
        self._cst = kwargs

        self._parametrization: pc.NamedTuple = None

        self._N = N
        self._J = J
        self._DIM_HD = DIM_HD

    @property
    def N(self):
        return self._N

    @property
    def J(self):
        return self._J

    @property
    def DIM_HD(self):
        return self._DIM_HD

    @property
    def DIM_LD(self):
        return self._parametrization.size - self._DIM_HD

    @property
    def parametrization(self):
        return self._parametrization

    @property
    def params_names(self):
        idx_params = self._parametrization.idx_params
        rep_num = [idx.stop - idx.start for idx in idx_params]
        repeat_name = np.repeat(idx_params._fields, rep_num)

        index_rep = np.concatenate(
            [[""] if n == 1 else [i for i in range(n)] for n in rep_num]
        )
        return np.char.add(repeat_name, index_rep)

    def reals1d_to_hstack_params(self, theta_reals1d):
        return jnp.hstack(list(self._parametrization.reals1d_to_params(theta_reals1d)))

    def new_params(self, **kwargs):
        theta_reals1d = self._parametrization.params_to_reals1d(**kwargs)
        return self._parametrization.reals1d_to_params(theta_reals1d)

    # ============================================================== #
    @abstractmethod
    def log_baseline_hazard(
        self,
        times,  # shape = (N,num)
        *args,
        **kwargs,
    ):
        """Function that return the log of the baseline hazard"""

    @abstractmethod
    def mixed_effect_function(self, *args, **kwargs):
        """Function that return an non linear fct that define the mixed effect models"""

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def link_function(
        self,
        alpha: jnp.ndarray,  # shape = (1,),
        times: jnp.ndarray,  # shape = (N,num)
        **kwargs,
    ):
        logistic_value = self.mixed_effect_function(times, **kwargs)
        assert logistic_value.shape == times.shape
        return alpha * logistic_value

    @functools.partial(jit, static_argnums=0)
    def log_hazard(
        self,
        times: jnp.ndarray,  # shape = (N,num)
        beta: jnp.ndarray,  # shape = (p,)
        alpha: jnp.ndarray,  # shape = (1,)
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N, num)
        """hazard(t) = h0(t) * exp(beta^T U  + alpha*m(t))
        with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

        return : log(b/a) + (b-1)*log(t/a) + beta^T U + alpha*m(t)
        """
        link_values = self.link_function(alpha, times, **kwargs)
        h0_values = self.log_baseline_hazard(times, **kwargs)

        beta_prod_cov = (cov @ beta)[:, None]
        assert beta_prod_cov.shape[0] == times.shape[0]

        return beta_prod_cov + h0_values + link_values

    @functools.partial(jit, static_argnums=0)
    def likelihood_survival_without_prior(
        self, params, T, delta, cov, **kwargs
    ) -> jnp.ndarray:
        """return likelihood without the gaussian prior"""
        (N,) = T.shape
        (p,) = params.beta.shape
        assert T.shape == (N,)
        assert delta.shape == (N,)
        # assert phi1.shape == (N,)
        # assert phi2.shape == (N,)
        assert cov.shape == (N, p)
        # ===================== #
        # === survival_likelihood === #
        # ===================== #
        # survival_likelihood = log(survival_fct) + log(hazard_fct)

        # ================= survival_fct ================= #
        # log_survival_fct = - int_0^T hazard(s) ds
        times = jnp.linspace(0, T, num=100)[1:].T

        hazard_kwargs = {
            "times": times,
            "mu3": params.mu3,
            "alpha": params.alpha,
            "beta": params.beta,
            "cov": cov,
        }
        log_hazard_value = self.log_hazard(**hazard_kwargs, **self._cst, **kwargs)
        assert times.shape == log_hazard_value.shape

        log_survival_fct = -jnp.trapz(jnp.exp(log_hazard_value), times)
        assert log_survival_fct.shape == (N,)
        # =============== end survival_fct =============== #

        # ================= hazard_fct ================= #
        # log_hazard_fct = delta * log(b*a^-b * T^{b-1}) + beta^T U + alpha*m(T, phi_g)
        # Comme times[:,-1] == T, on peut faire :
        log_hazard_fct = log_hazard_value[:, -1]
        assert log_hazard_fct.shape == (N,)
        # =============== end hazard_fct =============== #

        return delta * log_hazard_fct + log_survival_fct

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_mem_without_prior(
        self, params, Y, mem_obs_time, **kwargs
    ) -> jnp.ndarray:
        """return likelihood without the gaussian prior"""
        N, J = Y.shape
        assert mem_obs_time.shape == (N, J)
        # assert phi1.shape == (N,)
        # assert phi2.shape == (N,)

        pred = self.mixed_effect_function(
            mem_obs_time, **self._cst, **kwargs
        )  # shape = (N,J)

        likelihood_mem = -J / 2 * jnp.log(2 * jnp.pi * params.sigma2) - jnp.nansum(
            (Y - pred) ** 2, axis=1
        ) / (2 * params.sigma2)

        assert likelihood_mem.shape == (N,)
        return likelihood_mem

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def likelihood_array(self, theta_reals1d, **kwargs):
        return jnp.zeros(shape=(2,))

    @functools.partial(jit, static_argnums=0)
    def likelihood(self, theta_reals1d, **kwargs):
        return self.likelihood_array(theta_reals1d, **kwargs).sum()

    @functools.partial(jit, static_argnums=0)
    def jac_likelihood(self, theta_reals1d, **kwargs):
        return jacfwd(self.likelihood_array)(theta_reals1d, **kwargs)


# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #
def cov_simulation(PRNGKey, min, max, shape):
    cov = jrd.uniform(PRNGKey, minval=min, maxval=max, shape=shape)
    cov = cov - cov.mean(axis=0)[None, :]
    cov = jnp.array(cov, dtype=jnp.float32)

    return cov


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

    return {}, sim
