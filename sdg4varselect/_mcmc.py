"""
MCMC Simulation Module with Gibbs Sampling.

This module implements a Markov Chain for Gibbs sampling in MCMC simulations.
It includes a `gibbs_sampler` function to generate samples in each step,
and an `MCMC` class to manage parameters and acceptance adjustments for optimized sampling.

Functions
---------
gibbs_sampler : Main function for generating a sample using Gibbs sampling.

Create by antoine.caillebotte@inrae.fr
"""

from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit
from jax import random as jrd

from sdg4varselect._chain import Chain
from sdg4varselect.exceptions import Sdg4vsNanError, Sdg4vsInfError


@partial(
    jit,
    static_argnums=(1, 3),
)
def gibbs_sampler(
    key,  # 0
    data_name,  # 1
    standard_deviation,  # 2
    loglikelihood: callable,
    current_score: jnp.ndarray,
    **kwargs,
):
    """Parameters
    ----------
    key : jax.random.PRNGKey
        A PRNG key, consumable by random functions used to sample possible values for the data
    data_name : str
        The name of the target variable in `kwargs` for updating during sampling.
    standard_deviation : float
        The standard deviation for the proposal distribution used in Metropolis-Hastings.
    loglikelihood : callable
        The log-likelihood function, which takes arguments from `kwargs`.
    current_score : jnp.ndarray
        the current value of the log-likelihood
    **kwargs : dict
        all data required for log-likelihood computation.

    Returns
    -------
    key : jax.random.PRNGKey
        Updated PRNG key for the next sampling step.
    out : jnp.ndarray
        The new sampled values for the specified data, after the acceptance-rejection step.
    nacceptance : int
        Number of accepted proposals in this sampling iteration.

    """

    shape = kwargs[data_name].shape  # shape = (N,)
    old_data = kwargs[data_name].copy()

    # === proposal value ===
    key_bis, key = jrd.split(key)
    # kwargs[data_name] += standard_deviation * jrd.normal(key_bis, shape=shape)
    proposal = old_data + standard_deviation * jrd.normal(key_bis, shape=shape)
    proposal_score = loglikelihood(**{**kwargs, data_name: proposal})  # shape = (N,)

    # choose the new value
    key_bis, key = jrd.split(key)
    log_u = jnp.log(jrd.uniform(key_bis, shape=shape))  # shape = (N,)
    rejected_id = proposal_score - current_score <= log_u

    out = rejected_id * old_data + (1 - rejected_id) * proposal
    nacceptance = out.size - rejected_id.sum()

    return key, out, nacceptance


class MCMC(Chain):
    """
    Markov Chain Monte Carlo (MCMC) class implementing Gibbs sampling.

    The MCMC class is designed for running Gibbs sampling in an MCMC simulation.
    It allows tracking of acceptance rates, adaptively adjusting the proposal distribution's
    standard deviation, and calculating acceptance rates for improved performance in MCMC chains.

    Attributes
    ----------
    likelihood : callable
        The likelihood function used in the Gibbs sampler.
    adaptative_sd : bool
        A flag indicating whether to use adaptive standard deviation adjustments.
    lbd : float
        Adaptive parameter to adjust the standard deviation based on acceptance rate.
    sd : list of floats
        Standard deviation values used in the proposal distribution.

    Methods
    -------
    reset():
        Resets the MCMC chain to its initial state, including acceptance and standard deviation values.
    sample(key, theta_reals1d, size=1, **kwargs) -> list[np.ndarray]:
        Generates a specified number of samples using the Gibbs sampling process.
    gibbs_sampler_step(key, theta_reals1d, **kwargs):
        Executes a single Gibbs sampling step, updating the chain and adapting acceptance and standard deviation.
    acceptance_rate(i)
        Calculate the acceptance rate of the gibbs sampler iteration.
    """

    def __init__(self, likelihood: callable, sd: float = 1, **kwargs):
        """Constructor of MCMC chain with a gibbs sampler method."""
        super().__init__(**kwargs)

        self.__acceptance = [0.4]
        self.__sd = [sd]
        self.__adaptative_sd = False
        self.__lambda = 0.01
        self._likelihood: callable = likelihood

    def reset(self, x0: float = None) -> None:
        """Resets the MCMC to its initial state given when the object was created.

        Parameters
        ----------
            x0 : float, optional
                If None, x0 will be equal to the initial value of the MCMC.
        """
        super().reset(x0=x0)
        self.__acceptance = [self.__acceptance[0]]
        self.__sd = [self.__sd[0]]
        self.__lambda = 0.01

    def __repr__(self) -> str:
        out = super().__repr__()
        out += ", var = " + str(self._data.var()) + "]"

        return out

    @property
    def likelihood(self) -> callable:
        """returns likelihood

        Returns
        -------
        callable
            The likelihood function used in the Gibbs sampler.
        """
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood: callable):
        """Define the likelihood function used in the Gibbs sampler.

        Parameters
        ----------
            likelihood : callable
                The likelihood function used in the Gibbs sampler.
        """
        self._likelihood = likelihood

    @property
    def adaptative_sd(self) -> bool:
        """returns the boolean adaptative_sd
        Returns
        -------
        bool
            A flag indicating whether to use adaptive standard deviation adjustments
        """
        return self.__adaptative_sd

    @adaptative_sd.setter
    def adaptative_sd(self, x: bool) -> None:
        self.__adaptative_sd = x

    @property
    def lbd(self) -> float:
        """Returns the adaptative parameter of the sd
        Returns
        -------
        float
            Adaptive parameter to adjust the standard deviation based on acceptance rate.
        """
        return self.__lambda

    @property
    def sd(self) -> list:
        """Returns the proposal variance
        Returns
        -------
        list
            Standard deviation values used in the proposal distribution.
        """
        return self.__sd

    def acceptance_rate(self, i: int = None) -> np.ndarray:
        """Calculate the acceptance rate of the gibbs sampler iteration.

        This function computes the acceptance rate of gibbs sampling.
        If an integer `i` is provided, the function calculates the rate for that specific iteration.
        If `i` is None, it returns an array of acceptance rates for all iterations.

        Parameters
        ----------
            i : int, optional
                The specific iteration index to calculate the acceptance rate for.
                If None, calculates rates for all iterations.

        Returns
        -------
        np.ndarray
            The acceptance rate(s) as a numpy array.

        Raises
        ------
        ValueError
            If `i` is neither None nor an integer.
        """
        n_iteration = len(self.__acceptance)

        if i is None:
            # print(self.__acceptance)
            rate = np.array(self.__acceptance, dtype=np.float64) / np.array(
                [self._size * (i + 1) for i in range(n_iteration)]
            )
        elif isinstance(i, int):
            rate = float(self.__acceptance[i]) / (self._size * n_iteration)
        else:
            raise ValueError("i must be None or an int")

        return rate

    def sampler_step(self, key, **kwargs):
        """Performs a single Gibbs sampling step, updates the chain, and adapts the acceptance and standard deviation.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample possible values for the chain
        **kwargs : dict
            all data required for log-likelihood computation.

        Returns
        -------
        key_out : jax.random.PRNGKey
            Updated PRNG key for subsequent sampling steps.

        Raises
        ------
        Sdg4vsNanError
            If Nan is detected in the loglikelihood computation.
        Sdg4vsInfError
            If Nan is detected in the loglikelihood computation.
        """

        current_score = self._likelihood(**kwargs)
        if jnp.isnan(current_score).any():
            raise Sdg4vsNanError(
                "Nan detected in the loglikelihood during Gibbs Sampling!"
            )

        if jnp.isinf(current_score).any():
            raise Sdg4vsInfError(
                "Inf detected in the loglikelihood during Gibbs Sampling!"
            )

        key_out, data, nacceptance = gibbs_sampler(
            key, self.name, self.__sd[-1], self._likelihood, current_score, **kwargs
        )

        if jnp.isnan(data).any():
            raise Sdg4vsNanError(f"Nan detected in {self.name} during simulation step!")

        if jnp.isinf(data).any():
            raise Sdg4vsInfError(f"Inf detected in {self.name} during simulation step!")

        self._data[:] = data
        self.update_chain()  # append the new data to the chain

        nacceptance += self.__acceptance[-1]
        self.__acceptance.append(nacceptance)

        self.__adapt_sd()

        return key_out

    def __adapt_sd(self) -> None:
        """
        updating the variance of the gibbs sampler proposal
        to obtain an adequate acceptance rate
        """
        if not self.__adaptative_sd:
            return None

        sd_prop = self.__sd[-1]
        # lissage
        rate = 0.2 * self.acceptance_rate(-1) + 0.8 * self.acceptance_rate(-2)
        sd_prop *= jnp.exp(self.__lambda * (rate - 0.4))
        self.__lambda *= 0.999
        self.__sd.append(sd_prop)
        return None

    def plot(self, fig=None, id_max=None):
        """Plot the Markov Chain, acceptance rate, and proposal standard deviation (if adaptive)

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            A matplotlib figure object to plot on. If None, a new figure is created.
        id_max : int, optional
            The maximum number of iterations to plot. If None, all available iterations are plotted.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure.
        axs : numpy.ndarray
            An array of the matplotlib axes.

        Notes
        -----
        - The top plot shows the values of the MCMC chain up to `id_max`.
        - The middle plot displays the acceptance rate over iterations.
        - If the standard deviation is adaptive, a third plot shows the
            evolution of the proposal standard deviation.
        """

        if id_max is None:
            id_max = len(self.chain)
        if len(self.sd) == 1:
            axs = fig.subplots(2, 1, sharex=True)
        else:
            axs = fig.subplots(3, 1, sharex=True)

        axs[0].set_title(label="MCMC of " + self.name)

        axs[0].plot(self.chain[:id_max])
        axs[0].set_ylabel("Chain")

        axs[1].plot(self.acceptance_rate()[:id_max])
        axs[1].set_ylabel("Acceptance rate")

        if len(self.sd) != 1:
            axs[2].plot(self.sd[:id_max])
            axs[2].set_ylabel("Proposal sd")

        axs[-1].set_xlabel("Iteration")

        return fig, axs
