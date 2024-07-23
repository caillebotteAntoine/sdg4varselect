"""
Module for Stochastic Gradient descent algorithm preconditioned by the fisher information matrix.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=E1101
import itertools

import jax.numpy as jnp

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel

from sdg4varselect.algo.gradient_descent_fim import (
    GradientDescentFIM as GD_FIM,
    GradientDescentFIMSettings,
    get_GDFIM_settings,
)
from sdg4varselect.algo.abstract.abstract_algo_mcmc import AbstractAlgoMCMC
from sdg4varselect.outputs import GDResults


class StochasticGradientDescentFIM(AbstractAlgoMCMC, GD_FIM):
    """Stochastic Gradient descent algorithm preconditioned by the fisher information matrix"""

    def __init__(
        self,
        prngkey,
        max_iter: int,
        settings: GradientDescentFIMSettings,
    ):
        GD_FIM.__init__(self, max_iter, settings)
        AbstractAlgoMCMC.__init__(self, prngkey)

    def get_log_likelihood_kwargs(self, data):
        """return all the needed data"""
        return data | self.latent_data

    def results_warper(self, model, data, results, chrono):
        """warp results"""
        out = GDResults.new_from_list(results, chrono)
        likelihood = self.likelihood_marginal(model, data, out.last_theta)

        return GDResults.compute_with_model(model, out, likelihood)

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        AbstractAlgoMCMC._initialize_algo(
            self, model, log_likelihood_kwargs, theta_reals1d
        )
        GD_FIM._initialize_algo(self, model, log_likelihood_kwargs, theta_reals1d)

    # ============================================================== #
    def algorithm(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ):
        """iterative algorithm"""

        jac_shape = model.jac_likelihood(theta_reals1d, **log_likelihood_kwargs).shape
        jac = jnp.zeros(shape=jac_shape)

        for step in itertools.count():

            # Simulation
            self._one_simulation(log_likelihood_kwargs, theta_reals1d)

            # Gradient descent
            (theta_reals1d, jac, fisher_info, grad_precond) = (
                self._one_gradient_descent(
                    model, log_likelihood_kwargs, theta_reals1d, jac, step
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
    from sdg4varselect.outputs import MultiRunRes
    from sdg4varselect.models.logistic_mixed_effect_model import (
        LogisticMixedEffectsModel,
    )
    import jax.random as jrd
    import sdg4varselect.plot as sdgplt

    myModel = LogisticMixedEffectsModel(N=1000, J=5)

    p_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
    )

    obs, _ = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

    algo_settings = get_GDFIM_settings(preheating=400, heating=600)

    def one_fit(theta0):
        params = myModel.parametrization.reals1d_to_params(theta0)

        algo = StochasticGradientDescentFIM(jrd.PRNGKey(0), 1000, algo_settings)
        # =================== MCMC configuration ==================== #
        algo.add_mcmc(
            float(params.mu1),
            sd=0.001,
            size=myModel.N,
            likelihood=myModel.likelihood_array,
            name="phi1",
        )
        algo.latent_variables["phi1"].adaptative_sd = True
        algo.add_mcmc(
            float(params.mu2),
            sd=2,
            size=myModel.N,
            likelihood=myModel.likelihood_array,
            name="phi2",
        )
        algo.latent_variables["phi2"].adaptative_sd = True
        # ==================== END configuration ==================== #
        return algo.fit(myModel, obs, theta0, ntry=5)

    res = []
    for i in range(10):
        theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(one_fit(theta0))
    res = MultiRunRes(res)

    sdgplt.plot(res, params_star=p_star, params_names=myModel.params_names)
    print(f"chrono = {res.chrono}")
