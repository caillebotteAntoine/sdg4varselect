import numpy as np
from numpy.random import uniform
from math import pi

from .chain import chain


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

    def prior(self, i, loglikelihood, theta, **kwargs) -> np.ndarray:
        # Computation of the current target distrubtion score
        gaussian_prior = -0.5 * (
            np.log(2 * pi * self.__variance)
            + pow(self._data[i] - self.__mean, 2) / self.__variance
        )

        out = gaussian_prior + loglikelihood(i, theta, **kwargs)
        return out

    def gibbs_sampler_step(self, loglikelihood, theta, **kwargs) -> None:
        old_data = self._data.copy()

        nacceptance = self.__acceptance[-1]
        current_score = [
            self.prior(i, loglikelihood, theta, **kwargs) for i in range(self._size)
        ]
        standard_deviation = self.__sd[-1]

        for i in range(len(self._data)):
            self._data[i] += standard_deviation * np.random.normal()

            proposal_score = self.prior(i, loglikelihood, theta, **kwargs)

            rd = uniform()  # random value between 0 and 1
            if (proposal_score > current_score[i]) or (
                np.log(rd) < proposal_score - current_score[i]
            ):
                nacceptance += 1  # acceptation
            else:
                self._data[i] = old_data[i]  # rejection

        self.update_chain()  # append the new data to the chain

        self.__acceptance.append(nacceptance)
        if self.__adaptative_sd:
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

    def logistic_curve(
        x, supremum: float, midpoint: float, growth_rate: float
    ) -> float:
        return supremum / (1 + np.exp(-(x - midpoint) / growth_rate))

    time = np.linspace(100, 1500, num=J)
    Y = np.array([logistic_curve(time, phi1[i], phi2[i], phi3[i]) for i in range(N)])
    print(Y.shape)

    import matplotlib.pyplot as plt

    plt.plot(time, Y.transpose())
    # plt.show()

    def loglikelihood(i: int, theta, phi1, phi2, phi3) -> float:
        # print(phi1)
        pred = logistic_curve(time, phi1[i], phi2[i], phi3[i])
        # print(Y[i])
        # print(pred)
        out = sum(Y[i] - pred) ** 2 / theta["sigma2"]
        return out

    beta1 = chain(theta0[0])
    gamma1 = chain(theta0[3])

    MCMC_phi1 = MCMC_chain(theta0[0], N, 10, beta1.data(), gamma1.data())
    beta1.data()[0] = 1
    print(MCMC_phi1)

    phi1 = MCMC_phi1.data()

    MCMC_phi1.gibbs_sampler_step(
        loglikelihood, {"sigma2": sigma2}, phi1=phi1, phi2=phi2, phi3=phi3
    )
    print(MCMC_phi1)
    print(phi1)

    print(beta1)
