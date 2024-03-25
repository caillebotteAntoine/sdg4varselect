"""
Module for abstract class AbstractCoxModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221

from abc import abstractmethod
import functools

import numpy as np


import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_high_dim_model import AbstractHDModel


@functools.partial(jit, static_argnums=0)
def bisection_method_step(fun, a, b):
    m = (a + b) / 2

    neg_id = fun(a) * fun(m) <= 0

    b = jnp.where(neg_id, m, b)
    a = jnp.where(1 - neg_id, m, a)
    return a, b


def bisection_method(fun, a, b, eps=1e-3):
    eps0 = jnp.abs(b - a)
    maxiter = int(jnp.log2(eps0 / eps).max())

    for _ in range(maxiter):
        a, b = bisection_method_step(fun, a, b)

    print(jnp.abs(a - b).mean())
    return a


class AbstractCoxModel(AbstractModel, AbstractHDModel):
    """define a cox model with an abstract baseline hazard"""

    def __init__(self, N, P, **kwargs):
        AbstractModel.__init__(self, N=N, **kwargs)
        AbstractHDModel.__init__(self, P=P, **kwargs)

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"ACoxM_N{self.N}"

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""
        AbstractHDModel.init_dim(self, self.parametrization_size)

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(
        self,
        params,
        times,  # shape = (N,num)
        **kwargs,
    ):
        """Function that return the log of the baseline hazard"""

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_hazard(
        self,
        params,
        times: jnp.ndarray,  # shape = (N,num)
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N, num)
        """hazard(t) = h0(t) * exp(beta^T U )
        with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

        return : log(b/a) + (b-1)*log(t/a) + beta^T U
        """
        h0_values = self.log_baseline_hazard(params, times, **kwargs)

        beta_prod_cov = (cov @ params.beta)[:, None]
        assert beta_prod_cov.shape[0] == times.shape[0]

        return beta_prod_cov + h0_values

    @functools.partial(jit, static_argnums=0)
    def likelihood_survival_without_prior(
        self, params, T, delta, cov, **kwargs
    ) -> jnp.ndarray:
        """return likelihood without the gaussian prior"""
        (N,) = T.shape
        (p,) = params.beta.shape
        assert T.shape == (N,)
        assert delta.shape == (N,)
        assert cov.shape == (N, p)
        # ===================== #
        # === survival_likelihood === #
        # ===================== #
        # survival_likelihood = log(survival_fct) + log(hazard_fct)

        # ================= survival_fct ================= #
        # log_survival_fct = - int_0^T hazard(s) ds
        times = jnp.linspace(0, T, num=100)[1:].T

        log_hazard_value = self.log_hazard(
            params, times=times, cov=cov, **self._cst, **kwargs
        )
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
    def likelihood_array(self, theta_reals1d, **kwargs):
        """return likelihood as array each component for each individuals"""
        params = self._parametrization.reals1d_to_params(theta_reals1d)

        return self.likelihood_survival_without_prior(params, **kwargs)

    # ============================================================== #
    def sample(
        self,
        params_star,
        prngkey,
        weibull_censoring_loc,
        **kwargs,
    ):
        """Sample one data set for the model

        For data generation we seek t such that : P(T <= t ) = U([0,1])         # = 1 - S(t)
                                            ie : S(t) = 1 - U([0,1])

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

        so we seek t such that : - int_0^t lbd(s) ds = log(1 - U([0,1]))
                            ie : int_0^t lbd(s) ds + log(1 - U([0,1])) = 0
        """
        (
            prngkey_cov,
            prngkey_uni,
            prngkey_weibull,
        ) = jrd.split(prngkey, num=3)

        obs, sim = {}, {}
        # === cox_weibull_simulation === #

        uni = jrd.uniform(prngkey_uni, shape=(self.N,))
        cov = cov_simulation(prngkey_cov, cov_min=-1, cov_max=1, shape=(self.N, self.P))

        tmp = [0 for i in range(self.N)]
        # for i in range(self.N):

        @jit
        def f(T):
            # T = jnp.array([t for i in range(self.N)])

            t_linspace = jnp.linspace(0, T, num=1000)[1:].T

            y = self.log_hazard(params_star, t_linspace, cov, **kwargs, **self._cst)
            rho = jnp.trapz(y=jnp.exp(y), x=t_linspace)

            return rho + jnp.log(1 - uni)

        tmp = bisection_method(
            f,
            a=jnp.zeros(shape=(self.N,)),
            b=weibull_censoring_loc + jnp.zeros(shape=(self.N,)),
            eps=1e-4,
        )

        # tmp[i] = brenth(f, a=0, b=1000, args=(rho[i], uni[i]))

        sim["T uncensored"] = jnp.array(tmp)
        obs["cov"] = cov

        # ============================================================== #
        censoring = jrd.weibull_min(
            prngkey_weibull,
            weibull_censoring_loc,
            self._cst["b"] if "b" in self._cst else params_star.b,
            shape=sim["T uncensored"].shape,
        )

        T = np.minimum(sim["T uncensored"], censoring)
        delta = sim["T uncensored"] < censoring

        obs.update({"T": T, "delta": delta})
        sim["C"] = censoring

        return obs, sim


# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #


def cov_simulation(prngkey, cov_min, cov_max, shape):
    cov = jrd.uniform(prngkey, minval=cov_min, maxval=cov_max, shape=shape)
    cov = cov - cov.mean(axis=0)[None, :]
    cov = jnp.array(cov, dtype=jnp.float32)

    return cov


# def cox_simulation(
#     params, prngkey, beta_prod_cov, log_baseline_fct, log_baseline_kwargs, link_kwargs
# ):
#     """
#     lbd(t) = baseline_fct(t) * exp(beta^T U + linkfct(alpha, t, ...))
#     """

#     (n_ind,) = beta_prod_cov.shape

#     tmp = [0 for i in range(n_ind)]
#     key, prngkey = jrd.split(prngkey, num=2)
#     # rem : c'est pas pratique si uni est très très petit le log explose ...
#     uni = jrd.uniform(key, shape=(n_ind,))

#     beta_prod_cov_ind = beta_prod_cov[0]
#     link_ind_args = link_kwargs.copy()
#     for i in range(n_ind):
#         beta_prod_cov_ind = beta_prod_cov[i]

#         # extract individual argument for the link function
#         for key, value in enumerate(link_kwargs):
#             if isinstance(value, (np.ndarray, jnp.ndarray)) and value.shape != ():
#                 link_ind_args[key] = value[i]
#             else:
#                 link_ind_args[key] = value

#         print(log_baseline_kwargs)
#         print(link_ind_args)

#         def f(t):
#             """
#             example if baseline_fct = lbd_0 = weibull and linkfct = m the logistic function

#             For data generation we seek t such that : P(T <= t ) = U([0,1])         # = 1 - S(t)
#                                                 ie : S(t) = 1 - U([0,1])

#             where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
#             where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
#                                         with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

#             so we seek t such that : - int_0^t lbd(s) ds = log(1 - U([0,1]))
#                                 ie : int_0^t lbd(s) ds + log(1 - U([0,1])) = 0      # f(t) = 0
#             """

#             def lbd(s):
#                 """baseline_fct * exp[ beta^T U + linkfct(alpha, M(t, ...)) ]"""
#                 log_lbd0 = log_baseline_fct(s, **log_baseline_kwargs, **link_ind_args)
#                 return jnp.exp(log_lbd0 + beta_prod_cov_ind)

#             t_linspace = jnp.linspace(0, t, num=100)
#             return jnp.trapz(y=lbd(t_linspace), x=t_linspace) + jnp.log(1 - uni[i])

#         f = jnp.vectorize(f)

#         # Find a root of a function in the interval [a,b]
#         tmp[i] = brenth(f, a=0, b=1000)

#     sim = {"T uncensored": jnp.array(tmp)}

#     return {}, sim
