"""
Module for abstract Cox Model classes.

This module defines an abstract Cox model class.

Created by antoine.caillebotte@inrae.fr
"""

# pylint: disable = all

from abc import abstractmethod
import functools


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
from jax.scipy import integrate

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_high_dim_model import AbstractHDModel


@functools.partial(jit, static_argnums=0)
def _bisection_method_step(fun, a, b):
    """
    Perform a single step of the bisection method.

    Tant que (b - a) > ε
        m ← (a + b) / 2
        Si (f(a)*f(m) ≤ 0) alors
        b ← m
        sinon
        a ← m
        Fin Si
    Fin Tant que

    Parameters
    ----------
    fun : function
        Function for which the root is being searched.
    a : array-like
        Start of the interval.
    b : array-like
        End of the interval.

    Returns
    -------
    tuple
        Updated interval (a, b) after one bisection step.
    """
    m = (a + b) / 2

    neg_id = fun(a) * fun(m) <= 0

    b = jnp.where(neg_id, m, b)
    a = jnp.where(1 - neg_id, m, a)
    return a, b


def _bisection_method(fun, a, b, eps=1e-3):
    """Find a root of the function `fun` using the bisection method.

    Parameters
    ----------
    fun : function
        Function for which the root is being searched.
    a : array-like
        Start of the interval.
    b : array-like
        End of the interval.
    eps : float, optional
        Desired precision of the result, by default 1e-3.

    Returns
    -------
    array-like
        Approximate root of the function.

    Raises
    ------
    Sdg4vsNanError
        If Nan is detected in the fun computation.
    """
    if not (fun(a) * fun(b) <= 0).any():
        print(fun(a), fun(b))
    assert (fun(a) * fun(b) <= 0).any()

    eps0 = jnp.abs(b - a)
    maxiter = int(jnp.log2(eps0 / eps).max())

    for _ in range(maxiter):
        # print(f"fun(a)={fun(a)}\n, fun(b)={fun(b)},\n a={a},\n b={b}")
        a, b = _bisection_method_step(fun, a, b)

    if (
        jnp.isinf(fun(a)).any
        or jnp.isinf(fun(b)).any
        or jnp.isnan(fun(a)).any
        or jnp.isnan(fun(b)).any
    ):
        print(Warning("Nan or inf is detected in fun during bisection method!"))

    precision = [jnp.abs(a - b).sum(), jnp.abs(fun(a)).mean(), jnp.abs(fun(b)).mean()]
    print(
        f"bisection_method precision  =  {precision[0]}, mean(|f(a)|) = {precision[1]}, mean(|f(b)|) = {precision[1]}"
    )
    return (a + b) / 2


class AbstractCoxModel(AbstractModel, AbstractHDModel):
    """Abstract class defining a Cox model with an abstract baseline hazard.

    Parameters
    ----------
    N : int
        Number of samples.
    P : int
        Number of high-dimensional parameters in the model.
    """

    def __init__(self, N, P, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractModel.__init__(self, N=N, **kwargs)

    def init(self):
        AbstractModel.init(self)
        self.parametrization_size = self._parametrization.size
        self._is_initialized = True

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, survival_int_range, **kwargs):
        """Calculate the log of the baseline hazard.

        Parameters
        ----------
        params : dict
            Parameters of the model.
        survival_int_range : jnp.ndarray
            Array of survival observation, shape (N, num).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        jnp.ndarray
            Log of the baseline hazard.
        """
        raise NotImplementedError

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_hazard(
        self,
        params,
        survival_int_range: jnp.ndarray,  # shape = (N,num)
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N, num)
        """Compute the log of the hazard function.

        hazard(t) = h0(t) * exp(beta^T U )
        with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}
        return : log(b/a) + (b-1)*log(t/a) + beta^T U

        Parameters
        ----------
        params : dict
            Model parameters.
        survival_int_range : jnp.ndarray
            Array of time points, shape (N, num).
        cov : jnp.ndarray
            Covariates matrix, shape (N, p).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Log hazard values, shape (N, num).
        """
        h0_values = self.log_baseline_hazard(params, survival_int_range, **kwargs)
        assert h0_values.shape == survival_int_range.shape

        beta_prod_cov = (cov @ params.beta)[:, None]
        assert beta_prod_cov.shape[0] == survival_int_range.shape[0]

        return beta_prod_cov + h0_values

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(self, theta_reals1d, **kwargs):
        """Compute the log likelihood for each individual.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Parameters used to the log-likelihood computation.
        T : jnp.ndarray
            Observed survival times, shape (N,).
        delta : jnp.ndarray
            Censoring indicator, shape (N,).
        cov : jnp.ndarray
            Covariates matrix, shape (N, p).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Log likelihood values per individual.
        """
        params = self._parametrization.reals1d_to_params(theta_reals1d)

        T = kwargs["T"]
        cov = kwargs["cov"]
        delta = kwargs["delta"]
        survival_int_range = kwargs["survival_int_range"]

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
        log_hazard_value = self.log_hazard(params, **self._cst, **kwargs)
        assert survival_int_range.shape == log_hazard_value.shape
        # assert not jnp.isnan(log_hazard_value).any()
        # assert not jnp.isinf(log_hazard_value).any()

        log_survival_fct = -integrate.trapezoid(
            jnp.exp(log_hazard_value), survival_int_range
        )
        assert log_survival_fct.shape == (N,)
        # assert jnp.isnan(log_survival_fct).any()
        # assert jnp.isinf(log_survival_fct).any()
        # =============== end survival_fct =============== #

        # ================= hazard_fct ================= #
        # log_hazard_fct = delta * log(b*a^-b * T^{b-1}) + beta^T U + alpha*m(T, phi_g)
        # Comme survival_int_range[:,-1] == T, on peut faire :
        log_hazard_fct = log_hazard_value[:, -1]
        assert log_hazard_fct.shape == (N,)
        # =============== end hazard_fct =============== #

        return jnp.where(delta, log_hazard_fct, 0) + log_survival_fct

    # ============================================================== #
    def auto_def_survival_int_range(self, T, **kwargs):
        """jnp.linspace(0, T, num=100)[1:].T"""
        return {"survival_int_range": jnp.linspace(0, T, num=100)[1:].T}, {}

    @abstractmethod
    def censoring_simulation(self, prngkey, T, params_star, **kwargs):
        """Simulate censoring times.

        Parameters
        ----------
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        T : jnp.ndarray
            Uncensored event times.
        params_star : dict
            parameter used to sample the model
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Simulated censoring times.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def covariates_simulation(self, prngkey, **kwargs) -> jnp.ndarray:
        """Simulate covariates for the model.

        Parameters
        ----------
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Simulated covariates matrix.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def test_simulation_intervalle(self, params_star, start, stop, num=1000):
        t = jnp.linspace(start, stop, num=num)
        values = self.log_baseline_hazard(params=params_star, times=t, **self._cst)

        if values[0] != -jnp.inf or values[-1] != jnp.inf:
            raise ValueError(
                f"0 or jnp.inf not found between start = {start} and stop = {stop}"
            )

        id_max = jnp.min(jnp.where(values == jnp.inf)[0])
        if id_max == len(t) - 1:
            Warning("not found max")

        id_min = jnp.max(jnp.where(values == -jnp.inf)[0])
        if id_max == 0:
            Warning("not found min")

        return jnp.array([t[id_min], t[id_max]])

    def sample(
        self, params_star, prngkey, simulation_intervalle, **kwargs
    ) -> tuple[dict, dict]:
        """Sample a dataset from the Cox model.

        Parameters
        ----------
        params_star : dict
            Model parameters for sampling.
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        simulation_intervalle : tuple
            Interval within which to simulate times.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
                - dict: Generated observations.
                - dict: Simulated variables.
        Notes
        -----

        For data generation we seek t such that : P(T <= t ) = U([0,1])         # = 1 - S(t)
                                            ie : S(t) = 1 - U([0,1])

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

        so we seek t such that : - int_0^t lbd(s) ds = log(1 - U([0,1]))
                            ie : int_0^t lbd(s) ds + log(1 - U([0,1])) = 0 = f(t)
        """
        (
            prngkey_uni_cov,
            prngkey_uni,
            prngkey_censoring,
        ) = jrd.split(prngkey, num=3)

        cov = self.covariates_simulation(prngkey_uni_cov, **kwargs)
        assert cov.shape == (self.N, self.P), "cov matrix must have the good shape !"

        obs, sim = {"cov": cov}, {}
        # === cox_weibull_simulation === #

        uni = jrd.uniform(prngkey_uni, shape=(self.N,))

        tmp = [0 for i in range(self.N)]

        @jit
        def f(T):
            # T = jnp.array([t for i in range(self.N)])

            t_linspace = jnp.linspace(0, T, num=1000)[1:].T

            y = self.log_hazard(params_star, t_linspace, cov, **kwargs, **self._cst)
            rho = integrate.trapezoid(y=jnp.exp(y), x=t_linspace)

            return jnp.exp(-rho) - (1 - uni)

        a = simulation_intervalle[0] + jnp.zeros(shape=(self.N,))
        b = simulation_intervalle[1] + jnp.zeros(shape=(self.N,))
        # print(f"bisection try :\nfun(a) = {f(a)},\nfun(b) = {f(b)}")

        tmp = _bisection_method(f, a, b, eps=1e-8)

        sim["T uncensored"] = jnp.array(tmp)

        # ============================================================== #
        censoring = self.censoring_simulation(
            prngkey_censoring, sim["T uncensored"], params_star, **kwargs
        )

        T = jnp.minimum(sim["T uncensored"], censoring)
        delta = sim["T uncensored"] < censoring

        obs.update({"T": T, "delta": delta})
        sim["C"] = censoring
        sim["T_searching_uni"] = uni
        sim["f_min_searching"] = f(tmp)

        survival_int_range, _ = self.auto_def_survival_int_range()
        obs.update(survival_int_range)

        return obs, sim
