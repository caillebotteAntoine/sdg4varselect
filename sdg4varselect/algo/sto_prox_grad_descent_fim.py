# pylint: disable=E1101
import itertools

from typing import Optional
import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel

from sdg4varselect.algo.gradient_descent_FIM import (
    GradientDescentFIM,
    GradientDescentFIMSettings,
)
from sdg4varselect.algo.sto_grad_descent_fim import StochasticGradientDescentFIM


from sdg4varselect.algo.stochastic_gradient_descent_utils import proximal_operator


class StochasticProximalGradientDescentFIM(StochasticGradientDescentFIM):
    def __init__(
        self,
        prngkey,
        max_iter: int,
        settings: GradientDescentFIMSettings,
        lbd: Optional[float] = None,
        alpha: Optional[float] = 1.0,
    ):
        StochasticGradientDescentFIM.__init__(self, prngkey, max_iter, settings)

        self._lbd = lbd
        self._alpha = alpha

    # ============================================================== #

    def _one_proximal_operator(self, theta_reals1d, step, hd_mask):
        if self._lbd is None:
            return theta_reals1d

        return proximal_operator(
            theta_reals1d,
            stepsize=self._step_size_grad(step),
            lbd=self._lbd,
            alpha=self._alpha,
            hd_mask=hd_mask,
        )

    def algorithm(
        self,
        model: type(AbstractModel),
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ):
        """iterative algorithm"""
        dim_theta = theta_reals1d.shape[0]

        hd_mask = jnp.arange(dim_theta) >= dim_theta - model.DIMCovCox

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

            # Proximal operator
            theta_reals1d = self._one_proximal_operator(
                theta_reals1d=theta_reals1d,
                step=step,
                hd_mask=hd_mask,
            )

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta or jac")
                break

            yield (theta_reals1d, fisher_info, grad_precond, jnp.nan)

            if step > self._heating and jnp.sqrt((grad_precond**2).sum()) < 1e-3:
                break


if __name__ == "__main__":

    from sdg4varselect.plot import plot_sample
    from sdg4varselect.models.logistic_mixed_effect_model import (
        LogisticMixedEffectsModel,
    )
    from sdg4varselect.models.wcox_mem_joint_model import (
        WeibullCoxMemJointModel,
    )

    myMemModel = LogisticMixedEffectsModel(N=100, J=5)
    myModel = WeibullCoxMemJointModel(myMemModel, P=10)

    my_params_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=110.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.DIMCovCox - 4,))]
        ),
    )

    myobs, mysim = myModel.sample(
        my_params_star, jrd.PRNGKey(0), weibull_censoring_loc=77
    )

    _, _ = plot_sample(myobs, mysim, my_params_star, censoring_loc=77, a=80, b=35)

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
        res.append(algo.fit(myModel, myobs, theta0, partial_fit=True))

    import sdg4varselect.plot as sdgplt

    sdgplt.plot_theta(res, 7, my_params_star, myModel.params_names)
    sdgplt.plot_theta_HD(res, 7, my_params_star, myModel.params_names)
