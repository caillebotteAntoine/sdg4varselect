from math import pi

import jax.numpy as jnp
import numpy as np
from jax import jit

from sdg4varselect.chain import chain


@jit
def logistic_curve(
    x, supremum: float, midpoint: float, growth_rate: float
) -> jnp.ndarray:
    return supremum / (1 + jnp.exp(-(x - midpoint) / growth_rate))


@jit
def gaussian_prior(data, mean, variance) -> jnp.ndarray:
    # Computation of the current target distrubtion score
    out = jnp.log(2 * pi * variance) + jnp.power(data - mean, 2) / variance
    return -out / 2


class MCMC_chain(chain):
    def __init__(
        self,
        x0: float,
        size: int,
        sd: float,
        mean: np.ndarray,
        variance: np.ndarray,
        name=None,
    ):
        """Constructor of MCMC chain with a gibbs sampler method."""
        super().__init__(x0, size, name, "mcmc")

        self.__acceptance = [0]
        self.__sd = [sd]
        self.__adaptative_sd = False
        self.__lambda = 0.01

        if not isinstance(mean, np.ndarray):
            raise TypeError("mean must be an np.ndarray")
        self.__mean = mean

        if not isinstance(variance, np.ndarray):
            raise TypeError("variance must be an np.ndarray")
        self.__variance = variance

    def adaptative_sd(self, x: bool) -> None:
        self.__adaptative_sd = x

    def sd(self) -> list:
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

    def prior(self, loglike_without_prior_array, theta, **kwargs) -> jnp.ndarray:
        # Computation of the current target distrubtion score
        out = gaussian_prior(self._data, self.__mean, self.__variance)
        out += loglike_without_prior_array(theta, **kwargs)

        return out

    def gibbs_sampler_step(self, loglike_without_prior_array, theta, **kwargs) -> None:
        old_data = self._data.copy()

        nacceptance = self.__acceptance[-1]
        standard_deviation = self.__sd[-1]
        current_score = self.prior(loglike_without_prior_array, theta, **kwargs)

        # === proposal value ===
        self._data += standard_deviation * np.random.normal(size=self._size)
        proposal_score = self.prior(loglike_without_prior_array, theta, **kwargs)
        # choose the new value
        rd = np.log(np.random.uniform(size=self._size))
        rejected_id = proposal_score - current_score <= rd

        self._data[rejected_id] = old_data[rejected_id]
        nacceptance += self._size - rejected_id.sum()

        self.update_chain()  # append the new data to the chain

        self.__acceptance.append(nacceptance)
        self.adapt_sd()

    def adapt_sd(self) -> None:
        """
        updating the variance of the gibbs sampler proposal
        to obtain an adequate acceptance rate
        """
        if not self.__adaptative_sd:
            return None

        sd_prop = self.__sd[-1]
        rate = self.acceptance_rate(-1)

        if 0.5 > rate or rate > 0.7:
            if rate < 0.6:
                sd_prop /= 1 + self.__lambda

            if rate > 0.6:
                sd_prop *= 1 + self.__lambda

            self.__lambda *= 0.999
        self.__sd.append(sd_prop)


if __name__ == "__main__":
    beta = np.array([200, 500, 150])
    gamma = np.array([40, 100])
    sigma2 = 100

    theta0 = np.array([300, 400, 100, 30, 30, 10])

    N = 5
    J = 4
    phi1 = np.random.normal(beta[0], np.sqrt(gamma[0]), N)
    phi2 = np.random.normal(beta[1], np.sqrt(gamma[1]), N)
    phi3 = np.array([beta[2] for i in range(N)])

    eps = np.random.normal(0, np.sqrt(sigma2), N * J)

    time = np.linspace(100, 1500, num=J)
    Y = np.array([logistic_curve(time, phi1[i], phi2[i], phi3[i]) for i in range(N)])
    print(Y.shape)

    import matplotlib.pyplot as plt

    plt.plot(time, Y.transpose())
    # plt.show()

    def partial_loglike(theta, Y, time, phi1, phi2, phi3) -> jnp.ndarray:
        pred = logistic_curve(time, phi1, phi2, phi3)
        out = jnp.sum(jnp.power(Y - pred, 2))
        return jnp.sum(-out / (2 * theta.sigma2))

    def partial_loss_array(theta, Y, time, phi1, phi2, phi3):
        out = [
            partial_loglike(theta, Y[i], time, phi1[i], phi2[i], phi3[i])
            for i in range(len(phi1))
        ]
        return jnp.array(out)

    beta1 = chain(theta0[0])
    gamma1 = chain(theta0[3])

    MCMC_phi1 = MCMC_chain(theta0[0], N, 10, beta1.data(), gamma1.data())
    beta1.data()[0] = 1
    print(MCMC_phi1)

    phi1 = MCMC_phi1.data()

    from miscellaneous import namedTheta, time_profiler

    MCMC_phi1.gibbs_sampler_step(
        partial_loss_array,
        namedTheta(sigma2=sigma2),
        time=time,
        Y=Y,
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
    )

    @time_profiler(nrun=2000)
    def gibbs():
        MCMC_phi1.gibbs_sampler_step(
            partial_loss_array,
            namedTheta(sigma2=sigma2),
            time=time,
            Y=Y,
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
        )

    gibbs()

    print(MCMC_phi1)
    print(phi1)

    print(beta1)
