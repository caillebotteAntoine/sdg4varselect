"""Logistic Mixed Effects Model Example Implementation."""

import functools

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import parametrization_cookbook.jax as pc


from sdg4varselect.models.abstract.abstract_mixed_effects_model import (
    AbstractMixedEffectsModel,
)
from sdg4varselect._doc_tools import inherit_docstring


def m(
    params, times, eta0, eta1, **kwargs  # pylint: disable=unused-argument
) -> jnp.ndarray:
    """Logistic growth model.
    m(t) = eta0 / (1 + exp(-(t - eta1) / tau))
    where:
        - eta0: asymptotic value
        - eta1: time of inflection point
        - tau: fixed effect parameter (scale)
    The model is defined for each individual i and time point t.
    The shape of the output is (N, J) where N is the number of individuals and J is the number of time points.
    The shape of eta0 and eta1 is (N,).
    The shape of times is (N, J).

    Parameters
    ----------
    params : NamedTuple
        Model parameters.
    times : jnp.ndarray
        Time points.
    eta0 : jnp.ndarray
        Asymptotic values.
    eta1 : jnp.ndarray
        Inflection points.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    jnp.ndarray
        matrix of longitudinal values for each individual and time point.
    """
    out = eta0[:, None] / (1 + jnp.exp(-(times - eta1[:, None]) / params.tau))
    assert out.shape == times.shape
    return out


def dm(
    params, times, eta0, eta1, **kwargs  # pylint: disable=unused-argument
) -> jnp.ndarray:
    """Derivative of the logistic growth model with respect to time.
    dm/dt = (eta0 / tau) * exp(-(t - eta1) / tau) / (1 + exp(-(t - eta1) / tau))^2
    The shape of the output is (N, J) where N is the number of individuals and J is the number of time points.
    The shape of eta0 and eta1 is (N,).
    The shape of times is (N, J).
    Parameters
    ----------
    params : NamedTuple
        Model parameters.
    times : jnp.ndarray
        Time points.
    eta0 : jnp.ndarray
        Asymptotic values.
    eta1 : jnp.ndarray
        Inflection points.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    jnp.ndarray
        matrix of derivative values for each individual and time point.
    """
    e = jnp.exp(-(times - eta1[:, None]) / params.tau)
    out = eta0[:, None] * e / params.tau / (1 + e) ** 2
    assert out.shape == times.shape
    return out


@inherit_docstring
class LogisticMixedEffectsModel(AbstractMixedEffectsModel):
    """Example of a nonlinear mixed effects model.
    The model is a logistic growth model with two latent variables per individual:
        - eta0: the asymptotic value
        - eta1: the time of the inflection point
    The model is defined as:
        m(t) = eta0 / (1 + exp(-(t - eta1) / tau))
    where tau is a fixed effect parameter.
    The observations are assumed to be normally distributed around the model with a residual variance.
    """

    def __init__(self, N=1, J=1, **kwargs):
        AbstractMixedEffectsModel.__init__(self, N=N, J=J, **kwargs)
        self.add_latent_variables("eta0")
        self.add_latent_variables("eta1")

    @property
    def name(self) -> str:  # pylint: disable=missing-return-doc
        return f"LogisticGrowth_N{self.N}_J{self.J}"

    def init_parametrization(self):
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                mu_0=pc.Real(scale=1, loc=2),
                mu_1=pc.Real(scale=1, loc=10),
            ),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(1, 1)),
            tau=pc.RealPositive(scale=1),
            var_residual=pc.RealPositive(scale=1),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(  # pylint: disable=missing-return-doc
        self, params, *args, **kwargs
    ) -> jnp.ndarray:
        return m(params, *args, **kwargs)

    # ============================================================== #
    def sample(  # pylint: disable=missing-return-doc
        self, params_star, prngkey, **kwargs
    ) -> tuple[dict, dict]:
        (prngkey_mem, _) = jrd.split(prngkey, num=2)

        time = jnp.linspace(0, 30, self.J)
        time = jnp.tile(time, (self.N, 1))

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time
        )

        return {"mem_obs_time": time} | obs, sim


if __name__ == "__main__":

    import sdg4varselect.plotting as sdgplt

    myModel = LogisticMixedEffectsModel(N=100, J=10)

    p_star = myModel.new_params(
        mean_latent={"mu_0": 4, "mu_1": 15},
        cov_latent=jnp.diag(jnp.array([0.1, 1])),
        tau=4,
        var_residual=0.05,
    )

    myobs, mysim = myModel.sample(
        p_star,
        jrd.PRNGKey(1234),
    )

    # ploting longitudinal data
    fig = sdgplt.figure(7, 7)
    ax = fig.add_subplot(211)
    ax.plot(myobs["mem_obs_time"].T, myobs["Y"].T, "o-")
    ax.set_title("Longitudinal data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Y")
