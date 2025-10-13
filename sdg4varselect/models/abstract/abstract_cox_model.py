"""
Module for abstract Cox Model classes.

This module defines an abstract Cox model class.

Created by antoine.caillebotte@inrae.fr
"""

from abc import abstractmethod
import functools


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
from jax.scipy import integrate

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_high_dim_model import AbstractHDModel

from sdg4varselect._doc_tools import inherit_docstring


def _censoring_simulation(prngkey, T, c_percentage) -> jnp.ndarray:
    """Censoring simulation function.
    Parameters
    ----------
    prngkey : jax.random.PRNGKey
        random seed.
    T : jnp.ndarray
        Survival times.
    c_percentage : float
        Percentage of censoring.
    Returns
    -------
    jnp.ndarray
        Censored survival times."""
    prngkey_d, prngkey_c = jrd.split(prngkey)
    delta = jrd.bernoulli(prngkey_d, p=c_percentage, shape=T.shape)
    C = jrd.uniform(prngkey_c, minval=0, maxval=1, shape=T.shape) * T
    return jnp.where(delta, C, T + 1)


@inherit_docstring
class AbstractCoxModel(AbstractModel, AbstractHDModel):
    """Abstract class defining a Cox model with an abstract baseline hazard.

    Parameters
    ----------
    N : int
        Number of samples.
    P : int
        Number of high-dimensional parameters in the model.
    """

    def __init__(self, N, P=0, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractModel.__init__(self, N=N, **kwargs)

    def init(self):
        AbstractModel.init(self)
        self.parametrization_size = self._parametrization.size
        self._is_initialized = True

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, **kwargs) -> jnp.ndarray:
        """Calculate the log of the baseline hazard.

        Parameters
        ----------
        params : dict
            Parameters of the model.
        **kwargs : dict
            Additional parameters, containing survival_int_range the array of time points.


        Returns
        -------
        jnp.ndarray
            Log of the baseline hazard.
        """
        raise NotImplementedError

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def proportional_hazards_component(self, params, **kwargs) -> jnp.ndarray:
        """Compute the proportional hazards component.

        Parameters
        ----------
        params : dict
            Model parameters.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        jnp.ndarray
            Proportional hazards component.
        """
        raise NotImplementedError

    @functools.partial(jit, static_argnums=0)
    def log_hazard(self, params, **kwargs) -> jnp.ndarray:
        """Compute the log of the hazard function.

        hazard(t) = h0(t) * exp(phc)
        with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}
        return : log(b/a) + (b-1)*log(t/a) + phc
        e.g. in the cox model : phc = beta^T U or in the joint model : phc = beta^T U + alpha*m(t)

        Parameters
        ----------
        params : dict
            Model parameters.
        **kwargs : dict
            Additional parameters, containing survival_int_range the array of time points.

        Returns
        -------
        jnp.ndarray
            Log hazard values, shape (N, num).
        """
        survival_int_range = kwargs["survival_int_range"]
        h0_values = self.log_baseline_hazard(params, **kwargs)
        assert h0_values.shape == survival_int_range.shape

        phc = self.proportional_hazards_component(params, **kwargs)
        assert phc.shape[0] == survival_int_range.shape[0]
        assert phc.shape[1] == survival_int_range.shape[1] or phc.shape[1] == 1

        return phc + h0_values

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_likelihood_array(  # pylint: disable=missing-return-doc
        self, theta_reals1d, **kwargs
    ) -> jnp.ndarray:
        params = self._parametrization.reals1d_to_params(theta_reals1d)

        T = kwargs["T"]
        delta = kwargs["delta"]
        survival_int_range = kwargs["survival_int_range"]

        (N,) = T.shape
        assert T.shape == (N,)
        assert delta.shape == (N,)
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
    def auto_def_survival_int_range(
        self,
        T,
        **kwargs,  # pylint: disable=unused-argument
    ) -> jnp.ndarray:
        """Automatically define the survival interval range.

        Parameters
        ----------
        T : jnp.ndarray
            Uncensored event times.
        **kwargs : dict
            Additional parameters.
        Returns
        -------
        jnp.ndarray
            Survival interval range.
        """
        return jnp.linspace(0, T, num=100)[1:].T

    @abstractmethod
    def censoring_simulation(self, prngkey, T, params_star, **kwargs) -> jnp.ndarray:
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

    def sample(
        self, params_star, prngkey, linspace_num=100000, **kwargs
    ) -> tuple[dict, dict]:
        """Sample a dataset from the Cox model.

        Parameters
        ----------
        params_star : dict
            Model parameters for sampling.
        prngkey : jnp.ndarray
            Pseudo-random number generator key.
        linspace_num : int, optional
            Number of points in the linspace for survival interval range, by default 100000.
        **kwargs : dict
            Additional parameters.
            containing simulation_intervalle : tuple, Interval within which to simulate times.

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

        obs, sim = {}, {}

        obs["cov"] = self.covariates_simulation(prngkey_uni_cov, **kwargs, **self._cst)
        assert obs["cov"].shape == (
            self.N,
            self.P,
        ), "cov matrix must have the good shape !"

        # === cox_weibull_simulation === #
        t_linspace = jnp.tile(
            jnp.linspace(*kwargs["simulation_intervalle"], num=linspace_num)[1:],
            (self.N, 1),
        )
        pas = t_linspace[0, 1] - t_linspace[0, 0]

        # if x~exp(1) => exists u | F^{-1}(u) = x
        # ie u = 1 - exp(-x) => -log(1-u) = x
        sim["rexp"] = jrd.exponential(prngkey_uni, shape=(self.N,))
        log_h = self.log_hazard(
            params_star, survival_int_range=t_linspace, **obs, **self._cst, **kwargs
        )
        cumsum_h = jnp.cumsum(jnp.exp(log_h), axis=1) * pas
        # cumsum_h is increasing
        # argmax  return the index of the first maximum vaues.
        T = jnp.argmax(cumsum_h >= sim["rexp"][:, None], axis=1) * pas
        T = jnp.where(T == 0, jnp.nan, T)
        sim["T uncensored"] = jnp.where(
            jnp.isnan(T), kwargs["simulation_intervalle"][1], T
        )

        # ============================================================== #
        sim["C"] = self.censoring_simulation(
            prngkey_censoring, sim["T uncensored"], params_star, **kwargs, **self._cst
        )

        obs["T"] = jnp.minimum(sim["T uncensored"], sim["C"])
        obs["delta"] = jnp.logical_and(
            sim["T uncensored"] < sim["C"],
            sim["T uncensored"] != kwargs["simulation_intervalle"][1],
        )

        obs["survival_int_range"] = self.auto_def_survival_int_range(**obs)

        return obs, sim
