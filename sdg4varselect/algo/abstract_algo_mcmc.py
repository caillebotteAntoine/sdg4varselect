"""
Module for Abstract class containing common method for algorithm based on Monte-Carlo Markov Chains.

Create by antoine.caillebotte@inrae.fr
"""
# pylint: disable=C0116
from copy import deepcopy
from sdg4varselect._data_handler import DataHandler


class AbstractAlgoMCMC:
    """Abstract class containing common method for algorithm based on Monte-Carlo Markov Chains."""

    def __init__(self, prngkey, dh: DataHandler):
        self._data_handler = deepcopy(dh)
        self._prngkey = prngkey

    @property
    def data(self):
        return self._data_handler.data

    @property
    def latent_variables(self):
        return self._data_handler.latent_variables

    # ============================================================== #
    def add_mcmc(self, *args, **kwargs) -> None:
        """create a new mcmc chain and add it to the latent variable"""
        self._data_handler.add_mcmc(*args, **kwargs)

    # ============================================================== #
    def likelihood_marginal(self, model, prngkey, theta, size=1000):
        var_lat_sample = {}
        for var in self.latent_variables:
            var_lat_sample[var] = self.latent_variables[var].sample(
                prngkey,
                theta,
                size=size,
                **self.data,
            )

        likelihood_kwargs = deepcopy(self.data)
        out = [model.likelihood(theta, **likelihood_kwargs)]

        def add_val(k):
            for var in self.latent_variables:
                likelihood_kwargs[var] = var_lat_sample[var][k]

            new_val = model.likelihood(theta, **likelihood_kwargs)
            out.append(out[-1] + new_val)

        for k in range(1, 10):
            add_val(k)

        n_simu = 10
        while n_simu < size and abs(out[-2] / (n_simu - 1) - out[-1] / n_simu) >= 1e-3:
            add_val(n_simu)
            n_simu += 1
        return out[-1] / n_simu

    # ============================================================== #

    # @functools.partial(jit, static_argnums=0)
    def simulation(self, theta_reals1d):
        # Simulation
        for var_lat in self._data_handler.latent_variables.values():
            self._prngkey = var_lat.gibbs_sampler_step(
                self._prngkey, theta_reals1d, **self.data
            )

    # ============================================================== #


if __name__ == "__main__":
    pass
    # import jax.numpy as jnp

    # import jax.random as jrd
    # from sdg4varselect.models.logistic_joint_model import (
    #     Logistic_JM,
    #     sample_one,
    # )

    # myModel = Logistic_JM(N=100, J=5, DIM_HD=10)

    # myDH = sample_one(jrd.PRNGKey(0), myModel, weibull_censoring_loc=2000)

    # algo = AbstractAlgoMCMC(jrd.PRNGKey(0), myDH)
    # # =================== MCMC configuration ==================== #
    # algo.add_mcmc(
    #     0.3,
    #     sd=0.001,
    #     size=myModel.N,
    #     likelihood=myModel.likelihood_array,
    #     name="phi1",
    # )
    # algo.latent_variables["phi1"].adaptative_sd = True
    # algo.add_mcmc(
    #     90,
    #     sd=2,
    #     size=myModel.N,
    #     likelihood=myModel.likelihood_array,
    #     name="phi2",
    # )
    # algo.latent_variables["phi2"].adaptative_sd = True

    # params_star = myModel.new_params(
    #     mu1=0.3,
    #     mu2=90.0,
    #     mu3=7.5,
    #     gamma2_1=0.0025,
    #     gamma2_2=20,
    #     sigma2=0.001,
    #     alpha=11.11,
    #     beta=jnp.concatenate(
    #         [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.DIM_HD - 4,))]
    #     ),
    # )

    # myTheta = myModel.parametrization.params_to_reals1d(params_star)
    # prngkey = jrd.PRNGKey(0)
    # size = 1000

    # # likelihood = [
    # #     algo.likelihood_marginal(myModel, jrd.PRNGKey(0), myTheta, size=size)
    # #     for size in [10 * i for i in range(100)]
    # # ]

    # import matplotlib.pyplot as plt

    # plt.plot(jnp.array(out) / jnp.arange(1, n_simu + 1))
    # # plt.plot([10 * i for i in range(100)], likelihood)
