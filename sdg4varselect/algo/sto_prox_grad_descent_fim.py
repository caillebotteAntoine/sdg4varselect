"""
Module for Stochastic Proximal Gradient descent algorithm preconditioned by the fisher information matrix.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=E1101
from typing import Optional
import jax.numpy as jnp
import jax.random as jrd

from sdg4varselect.models import AbstractModel

import sdg4varselect.algo.preconditioner as sdgprecond

from sdg4varselect.algo.sto_grad_descent_fim import (
    StochasticGradientDescentFIM as SGD_FIM,
)

from sdg4varselect.algo.stochastic_gradient_descent_utils import proximal_operator

from sdg4varselect.algo.gradient_descent_fim import (
    GradientDescentFIMSettings as GradFimSettings,
    get_gdfim_settings,
)


class StochasticProximalGradientDescentFIM(SGD_FIM):
    """Stochastic Proximal Gradient descent algorithm preconditioned by the fisher information matrix"""

    def __init__(
        self,
        prngkey,
        max_iter: int,
        settings: GradFimSettings,
        preconditioner: sdgprecond.AbstractPreconditioner,
        lbd: Optional[float] = None,
        alpha: Optional[float] = 1.0,
    ):
        SGD_FIM.__init__(self, prngkey, max_iter, settings, preconditioner)

        self._lbd = lbd
        self._alpha = alpha

        # initial algo parameter
        self._hd_mask = jnp.zeros(shape=(1,))

    def _initialize_algo(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        # if not isinstance(model, AbstractHDModel):
        #     raise ValueError("model must be a high dimensional one !")

        SGD_FIM._initialize_algo(self, model, log_likelihood_kwargs, theta_reals1d)

        self._hd_mask = model.hd_mask
        self._fisher_mask = jnp.invert(self._hd_mask)

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
        (theta_reals1d, grad_precond, preconditioner) = self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )

        # Proximal operator
        theta_reals1d = self._one_proximal_operator(
            theta_reals1d=theta_reals1d,
            step=step,
        )

        return (theta_reals1d, grad_precond, preconditioner)


def algo_factory(name):  # , preheating, heating, learning_rate):
    if name == "Fisher":
        algo_settings = get_gdfim_settings(
            preheating=3000, heating=3500, learning_rate=1e-8
        )
        preconditioner = sdgprecond.Fisher(list(algo_settings)[1:])

    elif name == "AdaGrad":
        algo_settings = [
            {
                "learning_rate": 1,
                "preheating": 0,
                "heating": 4500,
                "max": 1e-1,
            }
        ]

        preconditioner = sdgprecond.AdaGrad()
    elif name == "FisherAdaGrad":
        algo_settings = get_gdfim_settings(
            preheating=3000, heating=3500, learning_rate=1e-8
        )

        preconditioner = sdgprecond.FisherAdaGradPreconditionner(
            P=myModel.P, settings=list(algo_settings)[1:]
        )
    else:
        raise ValueError("algo name must be FisherAdaGrad, Fisher or AdaGrad")

    return algo_settings, preconditioner


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sdg4varselect.outputs import MultiRunRes
    from sdg4varselect.models.logistic_mixed_effect_model import (
        HDLogisticMixedEffectsModel,
    )
    import jax.random as jrd
    import sdg4varselect.plot as sdgplt

    myModel = HDLogisticMixedEffectsModel(N=100, J=15, P=30)

    p_star = myModel.new_params(
        mean_latent={"mu1": 100, "mu2": 1200},
        cov_latent=jnp.diag(jnp.array([50, 2000])),
        tau=150,
        var_residual=30,
        beta=jnp.concatenate(
            [jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]
        ),
    )

    obs, sim = myModel.sample(p_star, jrd.PRNGKey(0))

    plt.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

    def one_fit(i, algo_name):
        """one_fit for one theta0"""
        theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))

        algo_settings, preconditioner = algo_factory(algo_name)

        # theta0 = theta0.at[3].set(-0.709)
        # print(myModel.parametrization.reals1d_to_params(theta0))

        algo = StochasticProximalGradientDescentFIM(
            jrd.PRNGKey(0), 5000, algo_settings, preconditioner, lbd=1e-1
        )
        # =================== MCMC configuration ==================== #
        algo.init_mcmc(theta0, myModel, sd={"phi1": 5, "phi2": 50})

        for var_lat in algo.latent_variables.values():
            var_lat.adaptative_sd = True
        # ==================== END configuration ==================== #
        out = algo.fit(myModel, obs, theta0, ntry=5, partial_fit=True)

        if i < 1:
            for var_lat in algo.latent_variables.values():
                sdgplt.plot_mcmc(var_lat)

        return out

    resFisher = MultiRunRes([one_fit(i, "Fisher") for i in range(10)])
    id_to_plot = [0, 1, 2, 3, 6, 7]
    sdgplt.plot(
        resFisher,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
        id_to_plot=id_to_plot,
    )
    _ = sdgplt.plot_theta_hd(
        resFisher,
        dim_ld=myModel.DIM_LD,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
    )

    print(f"chrono = {resFisher.chrono}")

    resAdaGrad = MultiRunRes([one_fit(i, "AdaGrad") for i in range(10)])
    sdgplt.plot(
        resAdaGrad,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
        id_to_plot=id_to_plot,
    )
    _ = sdgplt.plot_theta_hd(
        resAdaGrad,
        dim_ld=myModel.DIM_LD,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
    )

    print(f"chrono = {resAdaGrad.chrono}")

    resFisherAdaGrad = MultiRunRes([one_fit(i, "FisherAdaGrad") for i in range(10)])
    sdgplt.plot(
        resFisherAdaGrad,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
        id_to_plot=id_to_plot,
    )
    _ = sdgplt.plot_theta_hd(
        resFisherAdaGrad,
        dim_ld=myModel.DIM_LD,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
    )

    print(f"chrono = {resFisherAdaGrad.chrono}")

    # multi_grad_theta = jnp.array([res.grad for res in res]).T
    # multi_grad_theta = multi_grad_theta[:, :200, :]

    # print(multi_grad_theta.shape)
    # _ = sdgplt._plot_theta(
    #     multi_grad_theta,
    #     id_to_plot=[0, 1, 2, 3, 6, 7],
    #     params_names=myModel.params_names,
    #     fig=sdgplt.figure(),
    # )
    # ============================================================== #

    from sdg4varselect.outputs import TestResults

    id_to_plot = jnp.array([1, 2, 3, 4, 7, 8, 9, 10, 11]) - 1
    r = TestResults(
        [resFisher, resAdaGrad, resFisherAdaGrad],
        test_config=[
            {"name": "Fisher"},
            {"name": "AdaGrad"},
            {"name": "resFisherAdaGrad"},
        ],
    )
    scenarios_labels = ["Fisher", "AdaGrad", "resFisherAdaGrad"]
    x = r.last_theta[:, :, id_to_plot]
    fig = sdgplt.boxplot_estimation(
        x=x.T,
        hline=myModel.hstack_params(p_star)[id_to_plot],
        xlabels=scenarios_labels,
        title=myModel.params_names[id_to_plot],
        nrows=3,
        ncols=3,
        fig=sdgplt.figure(height=7, width=8),
    )
    fig.tight_layout()

    # ============================================================== #

    from matplotlib.gridspec import GridSpec

    G = GridSpec(len(r), 3)
    fig = sdgplt.figure()

    params_star_hd = p_star.beta
    non_zero = 3

    theta = r.last_theta[:, :, myModel.DIM_LD :]

    xticks = jnp.arange(0, myModel.P) + 1
    for i in range(len(r)):
        ax = plt.subplot(G[i, 0])

        sdgplt.myBoxplot(
            ax=ax,
            x=theta[i][:, :non_zero].T,
            xlabels=[f"{k+1}" for k in range(non_zero)],
        )
        ax.plot(
            xticks[:non_zero] - 1, params_star_hd[:non_zero], "bs", label="true value"
        )

        ax.legend()
        ax.set_ylabel(scenarios_labels[i])
        # == == == == #
        ax = sdgplt.plt.subplot(G[i, 1:])  # , sharey=ax)
        tt = theta[i][:, non_zero:].T
        theta_nonan = tt[jnp.array([~jnp.isnan(xx).any() for xx in tt]), :]

        # sdgplt.myBoxplot(ax = ax, x = theta_nonan)
        points = sum(
            [
                [(i + non_zero, xx) for xx in theta_nonan[i] if xx != 0]
                for i in range(theta_nonan.shape[0])
                if jnp.abs(theta_nonan[i]).sum() != 0
            ],
            [],
        )

        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            facecolors="none",
            edgecolors="k",
        )
        ax.hlines(0, xmin=non_zero, xmax=theta_nonan.shape[0], colors="k")

        xticks_nonzero = [non_zero + 1] + [
            (x + 1) * 25 for x in range((theta_nonan.shape[0] + non_zero) // 25)
        ]

        ax.set_xticks(xticks_nonzero, [str(x) for x in xticks_nonzero])

    ax = plt.subplot(G[0, 0])
    ax.set_title(
        f"Estimation of the {non_zero} non-zero components of $\\beta$", fontsize=15
    )
    ax = plt.subplot(G[0, 1:])  # , sharey=ax)
    ax.set_title("Estimation of the remaining zero components of $\\beta$", fontsize=15)

    fig.set_figheight(5)
    fig.set_figwidth(15)

    # ============================================================== #
