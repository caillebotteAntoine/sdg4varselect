"""pharmacokinetic Mixed Effects Model Example Implementation."""

# pylint: disable=missing-return-doc

import functools

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc

from sdg4varselect.models.abstract.abstract_mixed_effects_model import (
    AbstractMixedEffectsModel,
)
from sdg4varselect._doc_tools import inherit_docstring


# pylint: disable=unused-argument
def m(params, times, eta_0, eta_1, eta_2, **kwargs) -> jnp.ndarray:
    """Two compartment pharmacokinetic model with first order absorption.
    C(t) = (D * ka) / (V * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
    where:
        - v: volume of distribution
        - ka: absorption rate constant
        - cl: clearance
        - ke: elimination rate constant (ke = Cl / V)
        - D: dose (fixed to 100 in this example)
    The model is defined for each individual i and time point t.
    The shape of the output is (N, J) where N is the number of individuals
    and J is the number of time points.
    The shape of eta0 and eta1 is (N,).
    The shape of times is (N, J).

    Parameters
    ----------
    params : pc.NamedTuple
        The parameters of the model.
    times : jnp.ndarray
        The time points at which to evaluate the model. Shape (N, J).
    eta_0 : jnp.ndarray
        The latent variable for the log of the volume of distribution. Shape (N,).
    eta_1 : jnp.ndarray
        The latent variable for the log of the absorption rate constant. Shape (N,).
    eta_2 : jnp.ndarray
        The latent variable for the log of the clearance. Shape (N,). Shape (N,).
    **kwargs : dict
        Additional arguments. Must contain 'D' the dose.

    Returns
    -------
    jnp.ndarray
        The model evaluated at the given time points. Shape (N, J).
    """
    D = kwargs.get("D")

    vi = jnp.exp(eta_0)
    kai = jnp.exp(eta_1)
    cli = jnp.exp(eta_2)
    kei = cli / vi
    cst = (D * kai) / (vi * (kai - kei))

    out = cst[:, None] * (
        jnp.exp(-kei[:, None] * times) - jnp.exp(-kai[:, None] * times)
    )

    assert out.shape == times.shape
    return out


@inherit_docstring
class PKMixedEffectsModel(AbstractMixedEffectsModel):
    """Example of a nonlinear mixed effects model.
    The model is a two compartment pharmacokinetic model with first order absorption
    with three latent variables per individual:
        - eta_0: log of the volume of distribution (V)
        - eta_1: log of the absorption rate constant (ka)
        - eta_2: log of the clearance (Cl)
    The model is defined as:
        C(t) = (D * ka) / (V * (ka - ke)) * (exp(-ke * t) - exp(-ka * t))
    where ke = Cl / V and D is the dose (fixed to 100 in this example).
    The observations are assumed to be normally distributed around the model
    with a residual variance.
    """

    def __init__(self, N=1, **kwargs):
        AbstractMixedEffectsModel.__init__(self, N=N, J=15, **kwargs)
        self.add_latent_variables("eta_0")
        self.add_latent_variables("eta_1")
        self.add_latent_variables("eta_2")

    @property
    def name(self) -> str:
        return f"Pharmacokinetic_N={self.N}_J{self.J}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                V=pc.Real(scale=1, loc=4),
                ka=pc.Real(scale=1, loc=3),
                Cl=pc.Real(scale=1, loc=2),
            ),
            cov_latent=pc.MatrixDiagPosDef(dim=3, scale=1),
            # cov_latent=pc.MatrixSymPosDef(dim=3, scale=1),
            var_residual=pc.RealPositive(scale=1),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(self, params, *args, **kwargs) -> jnp.ndarray:
        return m(params, *args, **kwargs)

    # ============================================================== #

    def sample(self, params_star, prngkey, **kwargs) -> tuple[dict, dict]:
        (_, prngkey_mem) = jrd.split(prngkey, num=2)

        time = jnp.array(
            [0.05, 0.2, 0.3, 0.4, 0.6, 0.9, 1.5, 2, 3, 5, 7, 10, 15, 22, 30]
        )
        self._j = time.shape[0]
        time = jnp.tile(time, (self.N, 1))

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time
        )

        return {"mem_obs_time": time} | obs, sim
