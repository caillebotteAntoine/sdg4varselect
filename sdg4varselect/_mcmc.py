"""
MCMC Simulation Module with Gibbs Sampling.

This module implements a Markov Chain for Gibbs sampling in MCMC simulations.
It includes a `gibbs_sampler` function to generate samples in each step,
and an `MCMC` class to manage parameters and acceptance adjustments for optimized sampling.

Functions
---------
gibbs_sampler : Main function for generating a sample using Gibbs sampling.
"""

from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import jit
from jax import random as jrd

from sdg4varselect._chain import Chain


@partial(
    jit,
    static_argnums=(1, 3),
)
def gibbs_sampler(
    key,  # 0
    data_name,  # 1
    standard_deviation,  # 2
    loglikelihood: callable,
    theta_reals1d: jnp.ndarray,
    **kwargs
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
        The log-likelihood function, which takes `theta_reals1d` and additional arguments from `kwargs`.
    theta_reals1d : jnp.ndarray
        Parameters used as input to `loglikelihood`.
    **kwargs : dict
        Additional arguments passed to `loglikelihood`, including the data to be sampled.

    Returns
    -------
    key : jax.random.PRNGKey
        Updated PRNG key for the next sampling step.
    out : jnp.ndarray
        The new sampled values for the specified data, after the acceptance-rejection step.
    nacceptance : int
        Number of accepted proposals in this sampling iteration.
    """

    shape = kwargs[data_name].shape
    old_data = kwargs[data_name].copy()

    current_score = loglikelihood(theta_reals1d, **kwargs)

    # === proposal value ===
    key_bis, key = jrd.split(key)
    kwargs[data_name] += standard_deviation * jrd.normal(key_bis, shape=shape)
    proposal_score = loglikelihood(theta_reals1d, **kwargs)

    # choose the new value
    key_bis, key = jrd.split(key)
    rd = jnp.log(jrd.uniform(key_bis, shape=shape))
    rejected_id = proposal_score - current_score <= rd

    out = rejected_id * old_data + (1 - rejected_id) * kwargs[data_name]
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

    def reset(self):
        super().reset()
        self.__acceptance = [self.__acceptance[0]]
        self.__sd = [self.__sd[0]]
        self.__lambda = 0.01

    def __repr__(self) -> str:
        out = super().__repr__()
        out += " [mean = " + str(self._data.mean())
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
        -------
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

    def sample(
        self, key, theta_reals1d: np.ndarray, size=1, **kwargs
    ) -> list[np.ndarray]:
        """Generates samples using the Gibbs sampling process with a specified number of iterations.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample possible values for the chain
        theta_reals1d : array-like
            Parameters passed to the log-likelihood function used in the Gibbs sampler.
        size : int, optional
            Number of samples to generate (default is 1).
        **kwargs : dict
            Additional arguments to be passed to the `gibbs_sampler` function,
            such as data needed for the log-likelihood calculation.

        Returns
        -------
        list[np.ndarray]
            A list of sampled data arrays, where each entry corresponds to a new sample generated by the Gibbs sampler.
        """
        out = []
        for _ in range(size):
            key, data, _ = gibbs_sampler(
                key, self.name, self.__sd[-1], self._likelihood, theta_reals1d, **kwargs
            )

            out.append(data)
        return out

    def gibbs_sampler_step(self, key, theta_reals1d: np.ndarray, **kwargs):
        """Performs a single Gibbs sampling step, updates the chain, and adapts the acceptance and standard deviation.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample possible values for the chain
        theta_reals1d : array-like
            Parameters passed to the log-likelihood function used in the Gibbs sampler.
        **kwargs : dict
            Additional arguments passed to `gibbs_sampler`, such as data required for log-likelihood calculations.

        Returns
        -------
        key_out : jax.random.PRNGKey
            Updated PRNG key for subsequent sampling steps.
        """
        key_out, data, nacceptance = gibbs_sampler(
            key, self.name, self.__sd[-1], self._likelihood, theta_reals1d, **kwargs
        )

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
        rate = 0.02 * self.acceptance_rate(-1) + 0.98 * self.acceptance_rate(-2)

        if rate < 0.4:
            sd_prop /= 1 + self.__lambda

        else:  # if rate > 0.6:
            sd_prop *= 1 + self.__lambda

        # self.__lambda *= 0.999
        self.__sd.append(sd_prop)
        return None