from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import random as jrd

from sdg4varselect.miscellaneous import Chain


@partial(
    jit,
    static_argnums=(1, 3),
)
def gibbs_sampler(
    key,  # 0
    data_name,  # 1
    standard_deviation,  # 2
    loglikelihood,  # 3
    theta_reals1d,
    **kwargs
):
    key1, key2, key_out = jrd.split(key, num=3)

    shape = kwargs[data_name].shape
    old_data = kwargs[data_name].copy()

    current_score = loglikelihood(theta_reals1d, **kwargs)

    # === proposal value ===
    kwargs[data_name] += standard_deviation * jrd.normal(key1, shape=shape)
    proposal_score = loglikelihood(theta_reals1d, **kwargs)

    # choose the new value
    rd = jnp.log(jrd.uniform(key2, shape=shape))
    rejected_id = proposal_score - current_score <= rd

    out = rejected_id * old_data + (1 - rejected_id) * kwargs[data_name]
    nacceptance = out.size - rejected_id.sum()

    return key_out, out, nacceptance


class MCMC_chain(Chain):
    def __init__(
        self,
        x0: float,
        size: int,
        sd: float,
        likelihood,
        name: str,
    ):
        """Constructor of MCMC chain with a gibbs sampler method."""
        super().__init__(x0, size, name)

        self.__acceptance = [0]
        self.__sd = [sd]
        self.__adaptative_sd = False
        self.__lambda = 0.01
        self._likelihood = likelihood

    def reset(self):
        super().reset()
        self.__acceptance = [0]
        self.__sd = [self.__sd[0]]
        self.__lambda = 0.01

    def __repr__(self) -> str:
        out = super().__repr__()
        out += " [mean = " + str(self._data.mean())
        out += ", var = " + str(self._data.var()) + "]"

        return out

    @property
    def likelihood(self):
        """returns theta likelihood"""
        return self._likelihood

    @likelihood.setter
    def likelihood(self, likelihood):
        """Define the likelihood"""
        self._likelihood = likelihood

    @property
    def adaptative_sd(self) -> bool:
        """returns the boolean adaptative_sd"""
        return self.__adaptative_sd

    @adaptative_sd.setter
    def adaptative_sd(self, x: bool) -> None:
        self.__adaptative_sd = x

    @property
    def sd(self) -> list:
        """returns the proposal variance"""
        return self.__sd

    def acceptance_rate(self, i=None) -> np.ndarray:
        """compute the acceptance rate"""
        n_iteration = len(self.__acceptance)

        if i is None:
            # print(self.__acceptance)
            rate = np.array(self.__acceptance, dtype=np.float64) / np.array(
                [self._size * (i + 1) for i in range(n_iteration)]
            )
        elif isinstance(i, int):
            rate = float(self.__acceptance[i]) / (self._size * n_iteration)

        return rate

    def sample(self, key, theta_reals1d, size=1, **kwargs):
        out = []
        for _ in range(size):
            key, data, nacceptance = gibbs_sampler(
                key, self.name, self.__sd[-1], self._likelihood, theta_reals1d, **kwargs
            )

            out.append(data)
        return out

    def gibbs_sampler_step(self, key, theta_reals1d, **kwargs):
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
        rate = self.acceptance_rate(-1)

        if rate < 0.6:
            sd_prop /= 1 + self.__lambda

        if rate > 0.6:
            sd_prop *= 1 + self.__lambda

        self.__lambda *= 0.999
        self.__sd.append(sd_prop)


if __name__ == "__main__":
    pass

    # def f(x, mean, var):
    #     return 1 / np.sqrt(2 * np.pi * var**2) * ((x - mean) ** 2).mean()

    # x = np.random.normal(4, 0.5, 100)

    # mean = MCMC_chain(1.0, sd=1, size=1, name="mean")

    # key = jrd.PRNGKey(0)

    # def gibbs(key):
    #     for i in range(200):
    #         key = mean.gibbs_sampler_step(
    #             key,
    #             f,
    #             theta0_reals1d,
    #             **sim,
    #         )

    # gibbs(jrd.PRNGKey(0))

    # print(phi1.data().mean())
    # print(phi1.data().var())

    # from sdg4varselect.logistic_model import (
    #     likelihood_array,
    #     model,
    #     parametrization,
    #     theta0_reals1d,
    # )

    # key = jrd.PRNGKey(0)

    # theta0_params = parametrization.reals1d_to_params(theta0_reals1d)

    # print(theta0_reals1d)
    # print(theta0_params)

    # # ==== Data simulation ==== #
    # N, J = 10, 20

    # eps = np.random.normal(0, np.sqrt(100), (N, J))
    # sim = {
    #     "time": np.linspace(100, 1500, num=J),
    #     "phi1": np.random.normal(400, np.sqrt(40), N),
    #     "phi2": np.random.normal(500, np.sqrt(100), N),
    #     "phi3": np.array([150 for i in range(N)]),
    # }

    # print(sim["phi1"].mean())
    # print(sim["phi1"].var())

    # sim["Y"] = model(**sim) + eps

    # phi1 = MCMC_chain(float(theta0_params.beta1), sd=20, size=N, name="phi1")
    # sim["phi1"] = phi1.data()

    # from miscellaneous import time_profiler

    # print(likelihood_array(theta0_reals1d, **sim))

    # @time_profiler(nrun=10)
    # def gibbs(key):
    #     for i in range(200):
    #         key = phi1.gibbs_sampler_step(
    #             key,
    #             likelihood_array,
    #             theta0_reals1d,
    #             **sim,
    #         )

    # gibbs(jrd.PRNGKey(0))

    # print(phi1.data().mean())
    # print(phi1.data().var())

    # plot_mcmc(phi1)
