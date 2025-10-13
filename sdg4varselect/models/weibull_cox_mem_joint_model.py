"""
Joint Models with Weibull and Constant Hazards Implementation
"""

# pylint: disable=missing-return-doc

import jax.numpy as jnp
import jax.random as jrd

import matplotlib.pyplot as plt

from sdg4varselect.models import AbstractMixedEffectsModel
from sdg4varselect.models.abstract.abstract_cox_mem_joint_model import (
    AbstractJMvalueLink,
)

from sdg4varselect.models import LogisticMixedEffectsModel as NLMEM
from sdg4varselect.models.cox_models import WeibullCoxModel
from sdg4varselect.models.cox_models import CstCoxModel
from sdg4varselect._doc_tools import inherit_docstring


def plot_sample(obs, sim, title=None):
    """Plotting function for joint model samples.

    Parameters
    ----------
    obs : dict
        Observed data dictionary.
    sim : dict
        Simulated data dictionary.
    title : str, optional
        Title for the plot, by default None.

    Returns
    -------
    plot_fig : plt.Figure
        Matplotlib figure object.
    ax : plt.Axes
        Matplotlib axes object.
    """

    plot_fig = plt.figure()
    plot_fig.set_size_inches(8, 8)

    # ploting longitudinal data
    ax = plot_fig.add_subplot(211)
    ax.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

    # ploting survival data
    ax = plot_fig.add_subplot(212)
    ax.hist(
        [obs["T"][obs["delta"]], sim["T uncensored"]],  # , sim["C"]],
        bins=20,
        density=False,
        label=["censored survival time", "survival time"],  # , "censuring time"],
    )

    ax.legend()

    if title is not None:
        plot_fig.suptitle(title)
    ax.set_title(f'Simulation with {int((1-obs["delta"].mean())*100)}% censored data')

    return plot_fig, ax


@inherit_docstring
class CstHazardJM(AbstractJMvalueLink):
    """Joint Model with Constant Hazard
    The hazard function is defined as:
        h(t) = h0 * exp(alpha * m_i(t) + X^T beta)
    where h0 > 0 is the constant baseline hazard,
    alpha is the association parameter,
    beta is the vector of regression coefficients,
    and m_i(t) is the value of the longitudinal process at time t for individual i.
    """

    def __init__(self, MixedEffectsModel: type[AbstractMixedEffectsModel], **kwargs):
        AbstractJMvalueLink.__init__(self, MixedEffectsModel, CstCoxModel, **kwargs)

    @property
    def name(self) -> str:
        return f"JM_{self._mem.name}_{self._cox.name}_N{self.N}_J{self.J}"


@inherit_docstring
class WeibullHazardJM(AbstractJMvalueLink):
    """Joint Model with Weibull Hazard
    The hazard function is defined as:
        h(t) = (b/a) * (t/a)^(b-1) * exp(alpha * m_i(t) + X^T beta)
    where a > 0 is the scale parameter, b > 0 is the shape parameter,
    alpha is the association parameter,
    beta is the vector of regression coefficients,
    and m_i(t) is the value of the longitudinal process at time t for individual i.
    """

    def __init__(self, MixedEffectsModel: type[AbstractMixedEffectsModel], **kwargs):
        AbstractJMvalueLink.__init__(self, MixedEffectsModel, WeibullCoxModel, **kwargs)

    @property
    def name(self) -> str:
        return f"JM_{self._mem.name}_{self._cox.name}_N{self.N}_J{self.J}"


if __name__ == "__main__":

    # Simulation of data from a joint model with a constant hazard
    myModel = CstHazardJM(NLMEM, N=1000, J=10, P=5, C_percentage=0.3)

    p_star = myModel.new_params(
        mean_latent={"mu_0": 4, "mu_1": 15},
        cov_latent=jnp.diag(jnp.array([0.1, 1])),
        tau=4,
        var_residual=0.05,
        alpha=1.5,
        h0=0.05,
        beta=jnp.concatenate(
            [
                jnp.array([4, 2, -2, -4]),
                jnp.zeros(shape=(myModel.P - 4,)),
            ]
        ),
    )

    # Sampling
    myobs, mysim = myModel.sample(
        p_star,
        jrd.PRNGKey(1234),
        simulation_intervalle=[0, 45],
    )
    # Plotting
    fig, _ = plot_sample(myobs, mysim)
    fig.suptitle("Joint Model with Constant Hazard")

    # Simulation of data from a joint model with a Weibull hazard

    myModel = WeibullHazardJM(NLMEM, N=1000, J=10, P=5, C_percentage=0.2)

    p_star = myModel.new_params(
        mean_latent={"mu_0": 4, "mu_1": 15},
        cov_latent=jnp.diag(jnp.array([0.1, 1])),
        tau=4,
        var_residual=0.05,
        alpha=1.5,
        a=40,
        b=3,
        beta=jnp.concatenate(
            [
                jnp.array([4, 2, -2, -4]),
                jnp.zeros(shape=(myModel.P - 4,)),
            ]
        ),
    )

    # Sampling
    myobs, mysim = myModel.sample(
        p_star,
        jrd.PRNGKey(1234),
        simulation_intervalle=[0, 45],
    )
    # Plotting
    fig, _ = plot_sample(myobs, mysim)
    fig.suptitle("Joint Model with Weibull Hazard")
