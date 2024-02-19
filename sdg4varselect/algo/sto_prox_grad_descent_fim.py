"""
Module for Stochastic Proximal Gradient descent algorithm preconditioned by the fisher information matrix.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=E1101
import itertools

from typing import Optional
import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel

from sdg4varselect.algo.gradient_descent_fim import GradientDescentFIMSettings

from sdg4varselect.algo.sto_grad_descent_fim import (
    StochasticGradientDescentFIM as SGD_FIM,
)


from sdg4varselect.algo.stochastic_gradient_descent_utils import proximal_operator


class StochasticProximalGradientDescentFIM(SGD_FIM):
    """Stochastic Proximal Gradient descent algorithm preconditioned by the fisher information matrix"""

    def __init__(
        self,
        prngkey,
        max_iter: int,
        settings: GradientDescentFIMSettings,
        lbd: Optional[float] = None,
        alpha: Optional[float] = 1.0,
    ):
        SGD_FIM.__init__(self, prngkey, max_iter, settings)

        self._lbd = lbd
        self._alpha = alpha

        # initial algo parameter
        self._hd_mask = jnp.zeros(shape=(1,))

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        SGD_FIM._initialize_algo(self, model, likelihood_kwargs, theta_reals1d)

        dim_theta = theta_reals1d.shape[0]

        self._hd_mask = jnp.arange(dim_theta) >= dim_theta - model.P

    # ============================================================== #

    def _one_proximal_operator(self, theta_reals1d, step):
        if self._lbd is None:
            return theta_reals1d

        return proximal_operator(
            theta_reals1d,
            stepsize=self._step_size_grad(step),
            lbd=self._lbd,
            alpha=self._alpha,
            hd_mask=self._hd_mask,
        )

    def algorithm(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ):
        """iterative algorithm"""

        for step in itertools.count():

            # Simulation
            self._one_simulation(likelihood_kwargs, theta_reals1d)

            # Gradient descent
            (theta_reals1d, self._jac, fisher_info, grad_precond) = (
                self._one_gradient_descent(
                    model, likelihood_kwargs, theta_reals1d, self._jac, step
                )
            )

            # Proximal operator
            theta_reals1d = self._one_proximal_operator(
                theta_reals1d=theta_reals1d,
                step=step,
            )

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta or jac")
                break

            yield (theta_reals1d, fisher_info, grad_precond, jnp.nan)

            if step > self._heating and jnp.sqrt((grad_precond**2).sum()) < 1e-3:
                break


if __name__ == "__main__":

    from sdg4varselect.algo.gradient_descent_fim import (
        get_GDFIM_settings,
    )
    from sdg4varselect.plot import plot_sample
    from sdg4varselect.outputs import MultiRunRes
    import sdg4varselect.plot as sdgplt
    from sdg4varselect.models.wcox_mem_joint_model import (
        create_logistic_weibull_jm,
    )

    myModel = create_logistic_weibull_jm(N=100, J=5, P=10)

    p_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=110.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=77)

    _, _ = plot_sample(myobs, mysim, p_star, censoring_loc=77, a=80, b=35)

    algo_settings = get_GDFIM_settings(preheating=400, heating=600)

    def one_fit(theta0):
        params = myModel.parametrization.reals1d_to_params(theta0)

        algo = StochasticProximalGradientDescentFIM(jrd.PRNGKey(0), 1000, algo_settings)
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
        return algo.fit(myModel, myobs, theta0, ntry=5)

    res = []
    for i in range(10):
        theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(one_fit(theta0))
    res = MultiRunRes(res)

    sdgplt.plot_theta(res, 7, p_star, myModel.params_names)
    sdgplt.plot_theta_hd(res, 7, p_star, myModel.params_names)
    print(f"chrono = {res.chrono}")
