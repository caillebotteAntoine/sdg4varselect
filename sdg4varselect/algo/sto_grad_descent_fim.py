# pylint: disable=E1101
import itertools

import jax.numpy as jnp

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel

from sdg4varselect.algo.gradient_descent_FIM import (
    GradientDescentFIM,
    GradientDescentFIMSettings,
)
from sdg4varselect.algo.abstract_algo_mcmc2 import AbstractAlgoMCMC
from sdg4varselect.outputs_new import GDResults


class StochasticGradientDescentFIM(AbstractAlgoMCMC, GradientDescentFIM):
    def __init__(
        self,
        prngkey,
        max_iter: int,
        settings: GradientDescentFIMSettings,
    ):
        GradientDescentFIM.__init__(self, max_iter, settings)
        AbstractAlgoMCMC.__init__(self, prngkey)

    def get_likelihood_kwargs(self, data):
        """return all the needed data"""
        return data | self.latent_data

    def results_warper(self, model, data, results, chrono):
        """warp results"""
        out = GDResults.new_from_list(results, chrono)
        likelihood = self.likelihood_marginal(model, data, out.last_theta)

        return GDResults.compute_with_model(model, out, likelihood)

    # ============================================================== #
    def algorithm(
        self,
        model: type(AbstractModel),
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ):
        """iterative algorithm"""

        jac_shape = model.jac_likelihood(theta_reals1d, **likelihood_kwargs).shape
        jac = jnp.zeros(shape=jac_shape)

        for step in itertools.count():

            # Simulation
            self._one_simulation(likelihood_kwargs, theta_reals1d)

            # Gradient descent
            (theta_reals1d, jac, fisher_info, grad_precond) = (
                self._one_gradient_descent(
                    model, likelihood_kwargs, theta_reals1d, jac, step
                )
            )

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta or jac")
                break

            yield (theta_reals1d, fisher_info, grad_precond, jnp.nan)

            if step > self._heating and jnp.sqrt((grad_precond**2).sum()) < 1e-3:
                break


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sdg4varselect.models.logistic_mixed_effect_model import (
        LogisticMixedEffectsModel,
    )
    import jax.random as jrd

    myModel = LogisticMixedEffectsModel(N=100, J=5)

    my_params_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
    )

    obs, _ = myModel.sample(my_params_star, jrd.PRNGKey(0))

    plt.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

    theta0 = 0.2 * jrd.normal(jrd.PRNGKey(0), shape=(myModel.parametrization.size,))

    algo_settings = GradientDescentFIM.GradientDescentFIMSettings(
        step_size_grad={
            "learning_rate": 1e-8,
            "preheating": 400,
            "heating": 600,
            "max": 0.9,
        },
        step_size_approx_sto={
            "learning_rate": 1e-8,
            "preheating": 400,
            "heating": None,
            "max": 1,
        },
        step_size_fisher={
            "learning_rate": 1e-8,
            "preheating": 400,
            "heating": None,
            "max": 0.9,
        },
    )

    algo = StochasticGradientDescentFIM(jrd.PRNGKey(0), 1000, algo_settings)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(my_params_star.mu1),
        sd=0.001,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="phi1",
    )
    algo.latent_variables["phi1"].adaptative_sd = True
    algo.add_mcmc(
        float(my_params_star.mu2),
        sd=2,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="phi2",
    )
    algo.latent_variables["phi2"].adaptative_sd = True
    # ==================== END configuration ==================== #

    res = []
    for i in range(10):
        theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(algo.fit(myModel, obs, theta0, ntry=5))

    import sdg4varselect.plot as sdgplt

    sdgplt.plot_theta(res, 3, my_params_star, myModel.params_names)
