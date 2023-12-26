from sdg4varselect import jnp, jrd, learning_rate

from sdg4varselect.gradient import set_gradient_run_parameters, get_random_params0
from model_griesbach.model import jac_likelihood, likelihood, likelihood_array
from model_griesbach.sample import get_solver, sample


def estim_solver(solver, niter, kwargs_run_GD, verbatim=False, **run_parameters):
    solver.verbatim = verbatim

    solver = set_gradient_run_parameters(solver, **run_parameters)

    # ====================================================== #
    params = solver.params
    (DIM_COV_SURV,) = params.beta_surv.shape
    (DIM_COV_LONG,) = params.beta_long.shape
    fisher_mask = (
        jnp.arange(0, len(params) + DIM_COV_SURV + DIM_COV_LONG - 2) < len(params) - 3
    )  # -2 car on veut pas de 2 paramètre beta_surv, beta_long and alpha
    # ====================================================== #

    error_flag = False
    newrun = 0
    while newrun == 0 or (error_flag and newrun < 5):
        newrun += 1
        if error_flag:
            print(f"error detected try a new run ! ({newrun-1})")
            solver.reset_solver()

        res, error_flag = solver.stochastic_gradient(
            jac_likelihood=jac_likelihood,
            fisher_mask=fisher_mask,
            p=DIM_COV_SURV + DIM_COV_LONG,
            niter=niter,
            **kwargs_run_GD,
        )

    return res, solver, error_flag


def estim(
    data_set, params0, prng_key, niter, kwargs_run_GD, verbatim=False, **run_parameters
):
    solver, key = get_solver(prng_key, params0, data_set, likelihood, likelihood_array)

    res, solver, error_flag = estim_solver(
        solver, niter, kwargs_run_GD=kwargs_run_GD, verbatim=verbatim, **run_parameters
    )
    return res, solver, error_flag, key


if __name__ == "__main__":
    from sample import get_params_star
    from time import time

    DIM_COV = 4
    N_IND = 100
    params_star_stack, params_star, PRNGKey = get_params_star(
        jrd.PRNGKey(0), DIM_COV
    )  #

    params0_start = {
        "mu1": 0.5,
        "mu2": 0.5,
        "gamma2_1": 0.1,
        "gamma2_2": 0.1,
        "sigma2": 0.001,
        "alpha": 0.001,
        "beta_surv": jnp.ones(
            shape=(DIM_COV,)
        ),  # jrd.uniform(jrd.PRNGKey(0), shape=(DIM_COV,), minval=-1, maxval=1),
        "beta_long": -jnp.ones(
            shape=(DIM_COV,)
        ),  # jrd.uniform(jrd.PRNGKey(0), shape=(DIM_COV,), minval=-1, maxval=1),
    }

    params0_start["mu1"] = params_star.mu1
    params0_start["mu2"] = params_star.mu2
    params0_start["gamma2_1"] = params_star.gamma2_1
    params0_start["gamma2_2"] = params_star.gamma2_2
    params0_start["sigma2"] = params_star.sigma2
    params0_start["alpha"] = params_star.alpha
    params0_start["beta_long"] = params_star.beta_long
    params0_start["beta_surv"] = params_star.beta_surv

    data_set, sim, _ = sample(params_star, jrd.PRNGKey(0), N_IND=N_IND, J_OBS=5)

    ls = []
    lr = []
    for i in range(2):
        params0, PRNGKey = get_random_params0(
            PRNGKey, params0_start, error=0.2, uniform_on=["beta_surv", "beta_long"]
        )
        params0 = params0_start
        params0["beta_surv"] = jrd.uniform(
            PRNGKey, shape=(DIM_COV,), minval=-1, maxval=1
        )

        kwargs_run_GD = {
            "prox_regul": 0.5,
            "proximal_operator": False,
        }

        res, solver, error_flag, key = estim(
            data_set,
            params0,
            PRNGKey,
            niter=2000,
            kwargs_run_GD=kwargs_run_GD,
            verbatim=True,
            activate_fim=True,
            activate_jac_approx=True,
            lr=1e-8,
            # Grad
            plateau_grad=1000,
            plateau_grad_size=10000,
            scale_grad=1,
            # Jac
            plateau_jac=1000,
            plateau_jac_size=10000,
            scale_jac=1,
            # Fim
            plateau_fim=1000,
            plateau_fim_size=20000,
            scale_fim=0.9,
        )

        ls.append(solver)
        lr.append(res)

    from work import sdgplt
    import numpy as np

    fig, ax = sdgplt.plot_params(
        x=res.theta[:-1],
        x_star=np.array(params_star_stack),
        p=DIM_COV * 2,
        names=solver.params_names,
        logscale=False,
    )
    for i in range(len(lr) - 1):
        for k in range(lr[i].theta[:-1].shape[1] - DIM_COV * 2):
            ax[k].plot(lr[i].theta[:-1][:, k])

    beta = res.theta[:, 6:10]
    fig, ax = sdgplt.plot_params(
        x=beta,
        x_star=np.array(params_star_stack)[6:10],
        p=0,
        logscale=False,
    )
    for i in range(len(lr) - 1):
        beta = lr[i].theta[:, 6:10]
        for k in range(beta.shape[1]):
            ax[k].plot(beta[:, k])

    sdgplt.figure()
    _ = sdgplt.plot(np.array([res.likelihood for res in lr]).T)

    # _, _ = sdgplt.plot_params_hd(res.theta, p=2 * DIM_COV, location="right")

    # _, _ = sdgplt.plot_params_hd(res.grad, p=2 * DIM_COV, location="right")

    # for var in solver.latent_variables.values():
    #     sdgplt.plot_mcmc(var)

    theta = np.array([res.theta[-1] for res in lr])

    def plot_beta(theta, threshold=0):
        fig = sdgplt.figure()
        ax = fig.add_subplot(1, 1, 1)
        beta = theta[:, 6:]
        # beta_support = beta.sum(axis=0) != 0
        num_support = (beta != 0).sum(axis=0)
        print(num_support)

        id = np.array(
            [i for i in range(len(num_support)) if num_support[i] >= threshold]
        )
        xticks = [i + 1 for i in range(len(id))]
        #

        ax.boxplot(beta[:, id])
        ax.plot(xticks, params_star_stack[6:][id], "bs", label="true value")
        ax.set_xticks(xticks, id)
        ax.legend()

        return fig, ax

    plot_beta(theta)

    thetahat = solver.params
    print(solver.likelihood_marginal(1000, theta=solver.theta_reals1d))
    solver.theta_reals1d = params0_start
    print(solver.likelihood_marginal(1000, theta=solver.theta_reals1d))

    sdgplt.figure()
    _ = sdgplt.plot(
        np.exp(
            np.array(
                [
                    (solver.likelihood_kwargs["cov_long"] @ res.theta[i, 6:10])
                    for i in range(2000)
                ]
            )
        )
    )
