from sdg4varselect import jnp, jrd
from sdg4varselect.gradient import set_gradient_run_parameters

from joint_model_HELL.model import jac_likelihood, likelihood, likelihood_array
from joint_model_HELL.sample import get_solver, sample


def get_random_params0(prng_key, params0, error=0.2):
    p = params0.copy()
    for key in p:
        key_new, prng_key = jrd.split(prng_key, 2)
        p[key] *= float(jrd.uniform(key_new, minval=1.0 - error))

    key_new, prng_key = jrd.split(prng_key, 2)
    p["beta"] = jrd.uniform(prng_key, shape=p["beta"].shape, minval=-1, maxval=1)

    return p, key_new


def estim_solver(solver, niter, kwargs_run_GD, verbatim=False, **run_parameters):
    solver.verbatim = verbatim

    solver = set_gradient_run_parameters(solver, **run_parameters)

    # ====================================================== #
    params = solver.params
    (DIM_COV,) = params.beta.shape
    fisher_mask = (
        jnp.arange(0, len(params) + DIM_COV - 1) < len(params) - 2
    )  # -2 car on veut pas de 2 paramètre beta and alpha
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
            p=DIM_COV,
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

    DIM_COV = 5
    params0_start = {
        "mu1": 0.5,
        "mu2": 0.5,
        "gamma2_1": 0.1,
        "gamma2_2": 0.1,
        "sigma2": 0.001,
        "alpha": 0.001,
        "beta": jnp.zeros(shape=(DIM_COV,))
        # "beta": jrd.uniform(
        #     jrd.PRNGKey(int(time())), shape=(DIM_COV,), minval=-1, maxval=1
        # ),
    }

    params_star_stack, params_star_weibull, PRNGKey = get_params_star(
        jrd.PRNGKey(int(time())), DIM_COV
    )

    data_set, sim, PRNGKey = sample(
        params_star_weibull, PRNGKey, 100, 20, CENSORING=0.0
    )
    ls = []
    lr = []
    for i in range(10):
        params0, PRNGKey = get_random_params0(PRNGKey, params0_start)
        params0["beta"] = jnp.zeros(shape=(DIM_COV,))

        solver, PRNGKey = get_solver(
            jrd.PRNGKey(int(time())), params0, data_set, likelihood, likelihood_array
        )

        # print(data_set)

        kwargs_run_GD = {
            "prox_regul": 0.005,
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
            plateau_grad_size=200,
            scale_grad=1,
            # Jac
            plateau_jac=1000,
            plateau_jac_size=1000,
            scale_jac=1,
            # Fim
            plateau_fim=1000,
            plateau_fim_size=2000,
            scale_fim=0.9,
        )

        ls.append(solver)
        lr.append(res)

    from work import sdgplt
    import numpy as np

    _, _ = sdgplt.plot_params(
        x=res.theta[:-1],
        x_star=np.array(params_star_stack),
        p=0,  # DIM_COV * 2,
        names=solver.params_names,
        logscale=False,
    )

    _, _ = sdgplt.plot_params_hd(res.theta, p=2 * DIM_COV, location="right")

    _, _ = sdgplt.plot_params_hd(res.grad, p=2 * DIM_COV, location="right")

    # for var in solver.latent_variables.values():
    #     sdgplt.plot_mcmc(var)

    theta = np.array([res.theta[-1] for res in lr])

    def plot_beta(theta, threshold=0):
        fig = sdgplt.figure()
        ax = fig.add_subplot(1, 1, 1)
        beta = theta
        # beta_support = beta.sum(axis=0) != 0
        num_support = (beta != 0).sum(axis=0)
        print(num_support)

        id = np.array(
            [i for i in range(len(num_support)) if num_support[i] >= threshold]
        )
        xticks = [i + 1 for i in range(len(id))]
        #

        ax.boxplot(beta[:, id])
        ax.plot(xticks, params_star_stack[id], "bs", label="true value")
        ax.set_xticks(xticks, id)
        ax.legend()

        return fig, ax

    plot_beta(theta)

    cov = data_set["cov"]

    sdgplt.figure()
    _ = sdgplt.plot([cov @ beta[6:] for beta in res.theta])

    sdgplt.figure()
    _ = sdgplt.plot([beta[6:] for beta in res.theta])
