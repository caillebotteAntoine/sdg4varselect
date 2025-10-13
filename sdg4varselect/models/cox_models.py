"""Cox model with Weibull Hazard, Constant Hazard and Gompertz Hazard implementations"""

# pylint: disable=missing-return-doc

import functools

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc

from sdg4varselect.models.abstract.abstract_cox_model import (
    AbstractCoxModel,
    _censoring_simulation,
)
from sdg4varselect._doc_tools import inherit_docstring


@inherit_docstring
class _AbstractCoxModel(AbstractCoxModel):
    """Weibull Cox Model
    The hazard function is defined as:
        h(t) = (b/a) * (t/a)^(b-1) * exp(X^T beta)
    where a > 0 is the scale parameter, b > 0 is the shape parameter,
    and beta is the vector of regression coefficients.
    """

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def proportional_hazards_component(self, params, **kwargs) -> jnp.ndarray:
        return jnp.dot(kwargs["cov"], params.beta)[None, :].T

    # ============================================================== #
    def covariates_simulation(self, prngkey, **kwargs) -> jnp.ndarray:
        cov = jrd.bernoulli(prngkey, p=0.2, shape=(self.N, self.P))
        cov -= jnp.mean(cov, axis=0, keepdims=True)
        return cov

    # ============================================================== #
    def censoring_simulation(self, prngkey, T, params_star, **kwargs) -> jnp.ndarray:
        return _censoring_simulation(prngkey, T, kwargs["C_percentage"])


@inherit_docstring
class WeibullCoxModel(_AbstractCoxModel):
    """Weibull Cox Model
    The hazard function is defined as:
        h(t) = (b/a) * (t/a)^(b-1) * exp(X^T beta)
    where a > 0 is the scale parameter, b > 0 is the shape parameter,
    and beta is the vector of regression coefficients.
    """

    @property
    def name(self) -> str:
        return f"WeibullCox_N{self.N}P_{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            a=pc.RealPositive(scale=1),
            b=pc.RealPositive(scale=1),
            beta=pc.Real(shape=(self.P,), scale=1),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, **kwargs) -> jnp.ndarray:
        a = params.a
        b = params.b
        survival_int_range = kwargs["survival_int_range"]
        return jnp.log(b / a) + (b - 1) * jnp.log(survival_int_range / a)


@inherit_docstring
class CstCoxModel(_AbstractCoxModel):
    """Constant Cox Model
    The hazard function is defined as:
        h(t) = h0 * exp(X^T beta)
    where h0 > 0 is the constant baseline hazard,
    and beta is the vector of regression coefficients.
    """

    @property
    def name(self) -> str:
        return f"CstCox_N{self.N}P_{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            h0=pc.RealPositive(scale=0.1),
            beta=pc.Real(shape=(self.P,), scale=1),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, **kwargs) -> jnp.ndarray:
        survival_int_range = kwargs["survival_int_range"]
        return jnp.log(params.h0) * jnp.ones(survival_int_range.shape)


@inherit_docstring
class GompertzCoxModel(_AbstractCoxModel):
    """Gompertz Cox Model
    The hazard function is defined as:
        h(t) = (1/b) * exp(t/a) * exp(X^T beta)
    where a > 0 is the scale parameter, b > 0 is the shape parameter,
    and beta is the vector of regression coefficients.
    """

    @property
    def name(self) -> str:
        return f"GompertzCox_N{self.N}P_{self.P}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            a=pc.RealPositive(scale=10),
            b=pc.RealPositive(scale=100),
            beta=pc.Real(shape=(self.P,), scale=1),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(self, params, **kwargs) -> jnp.ndarray:
        a = params.a
        b = params.b
        survival_int_range = kwargs["survival_int_range"]
        return (survival_int_range / a) - jnp.log(b)
