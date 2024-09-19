"""
Module for Stochastic Gradient descent algorithm preconditioned by the fisher information matrix.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=E1101
import jax.numpy as jnp
from datetime import datetime

from sdg4varselect.learning_rate import LearningRate

from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_latent_variables_model import (
    log_likelihood_marginal,
)

from sdg4varselect.algo.gradient_descent_fim import (
    GradientDescentFIM as GD_FIM,
    # GradientDescentFIMSettings,
    # get_gdfim_settings,
)
from sdg4varselect.algo.abstract.abstract_algo_mcmc import AbstractAlgoMCMC
from sdg4varselect.outputs import GDResults

from sdg4varselect.algo.preconditioner import (
    Fisher,
    AdaGrad,
    AbstractPreconditioner,
)


class StochasticGradientDescentFIM(AbstractAlgoMCMC, GD_FIM):
    """Stochastic Gradient descent algorithm preconditioned by the fisher information matrix"""

    def __init__(
        self,
        prngkey,
        max_iter: int,
        step_size: LearningRate,
        preconditioner: AbstractPreconditioner,
        threshold=1e-4,
    ):
        GD_FIM.__init__(self, max_iter, step_size, preconditioner, threshold)
        AbstractAlgoMCMC.__init__(self, prngkey)
        self._pre_heating = 1000

    def get_log_likelihood_kwargs(self, data):
        """return all the needed data"""
        return data | self.latent_data

    def results_warper(self, model, data, results, chrono):
        """warp results"""
        chrono_start = datetime.now()

        out = GDResults.new_from_list(results, chrono)

        out.likelihood = log_likelihood_marginal(
            model, self._prngkey, data, out.last_theta
        )
        out.reals1d_to_hstack_params(model)
        out.chrono += datetime.now() - chrono_start
        return out

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

        for _ in range(self._pre_heating):
            self._one_simulation(log_likelihood_kwargs, theta_reals1d)

    # ============================================================== #
    def _algorithm_one_step(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        """one iterative algorithm step"""
        # Simulation
        self._one_simulation(log_likelihood_kwargs, theta_reals1d)

        # Gradient descent
        return self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )


# def algo_factory(
#     name, regularization=1, learning_rate=1e-8
# ):  # , preheating, heating, learning_rate):
#     if name == "Fisher":
#         algo_settings = get_gdfim_settings(
#             preheating=2000, heating=2500, learning_rate=learning_rate
#         )
#         preconditioner = Fisher(list(algo_settings)[1:])

#     elif name == "AdaGrad":
#         algo_settings = [
#             {
#                 "learning_rate": 1,
#                 "preheating": 0,
#                 "heating": 3500,
#                 "max": learning_rate,
#             }
#         ]

#         preconditioner = AdaGrad(regularization=regularization)
#     else:
#         raise ValueError("algo name must be Fisher or AdaGrad")

#     return algo_settings, preconditioner


# if __name__ == "__main__":

#     import matplotlib.pyplot as plt
#     from sdg4varselect.outputs import MultiRunRes
#     from sdg4varselect.models.examples.logistic_mixed_effect_model import (
#         LogisticMixedEffectsModel,
#     )
#     import jax.random as jrd
#     import sdg4varselect.plot as sdgplt

#     myModel = LogisticMixedEffectsModel(N=100, J=15)

#     p_star = myModel.new_params(
#         mean_latent={"mu1": 100, "mu2": 1200},
#         cov_latent=jnp.diag(jnp.array([20, 2000])),
#         tau=150,
#         var_residual=30,
#     )

#     obs, sim = myModel.sample(p_star, jrd.PRNGKey(0))

#     plt.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

#     def one_fit(i, algo_name, regularization=1, learning_rate=1e-3):
#         """one_fit for one theta0"""
#         theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))

#         algo_settings, preconditioner = algo_factory(
#             algo_name, regularization, learning_rate
#         )

#         algo = StochasticGradientDescentFIM(
#             jrd.PRNGKey(0),
#             4000,
#             algo_settings,
#             preconditioner,
#         )
#         # =================== MCMC configuration ==================== #
#         algo.init_mcmc(theta0, myModel, sd={"phi1": 5, "phi2": 50})

#         for var_lat in algo.latent_variables.values():
#             var_lat.adaptative_sd = True
#         # ==================== END configuration ==================== #
#         out = algo.fit(myModel, obs, theta0, ntry=5, partial_fit=True)

#         if i < -2:
#             for var_lat in algo.latent_variables.values():
#                 sdgplt.plot_mcmc(var_lat)

#         return out

#     # res = MultiRunRes([one_fit(i, "Fisher") for i in range(1)])

#     # sdgplt.plot(
#     #     res,
#     #     params_star=myModel.hstack_params(p_star),
#     #     params_names=myModel.params_names,
#     #     id_to_plot=[0, 1, 2, 3, 6, 7],
#     # )
#     # print(f"chrono = {res.chrono}")

#     regularization = [1, 1e-1, 1e-2, 1e-4, 1e-6]
#     res = MultiRunRes(
#         [one_fit(1, "AdaGrad", i, learning_rate=0.01) for i in regularization]
#     )
#     # res = MultiRunRes([one_fit(i, "AdaGrad", 1, 0.1) for i in range(1)])

#     p_emv = myModel.new_params(
#         mean_latent={"mu1": sim["phi1"].mean(), "mu2": sim["phi2"].mean()},
#         cov_latent=jnp.diag(jnp.array([sim["phi1"].var(), sim["phi2"].var()])),
#         tau=150,
#         var_residual=30,
#     )
#     sdgplt.plot(
#         res,
#         params_star=myModel.hstack_params(p_emv),
#         params_names=myModel.params_names,
#         id_to_plot=[0, 1, 2, 3, 6, 7],
#     )
#     print(f"chrono = {res.chrono}")

#     # multi_grad_theta = jnp.array([res.grad for res in res]).T
#     # multi_grad_theta = multi_grad_theta[:, :200, :]

#     # print(multi_grad_theta.shape)
#     # _ = sdgplt._plot_theta(
#     #     multi_grad_theta,
#     #     id_to_plot=[0, 1, 2, 3, 6, 7],
#     #     params_names=myModel.params_names,
#     #     fig=sdgplt.figure(),
#     # )
