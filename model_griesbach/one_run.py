from sdg4varselect import jnp, jrd, learning_rate

from model_griesbach.model import jac_likelihood, likelihood, likelihood_array
from model_griesbach.sample import get_solver, sample


def get_random_params0(prng_key, params0, error=0.2):
    p = params0.copy()
    for key in p:
        key_new, prng_key = jrd.split(prng_key, 2)
        p[key] *= float(jrd.uniform(key_new, minval=1.0 - error))

    key_new, prng_key = jrd.split(prng_key, 2)
    p["beta_surv"] = jrd.uniform(
        prng_key, shape=p["beta_surv"].shape, minval=-1, maxval=1
    )

    key_new, prng_key = jrd.split(prng_key, 2)
    p["beta_long"] = jrd.uniform(
        prng_key, shape=p["beta_long"].shape, minval=-1, maxval=1
    )

    return p, key_new


def set_solver_run_parameters(
    solver,
    activate_fim=False,
    activate_jac_approx=True,
    lr=1e-8,
    # Grad
    plateau_grad=400,
    scale_grad=1,
    plateau_grad_size=100,
    # Jac
    plateau_jac=300,
    plateau_jac_size=50,
    scale_jac=0.5,
    # Fim
    plateau_fim=750,
    plateau_fim_size=50,
    scale_fim=0.95,
):
    if activate_jac_approx:
        solver.step_size = learning_rate(
            plateau_jac,
            float(jnp.log(lr)),
            plateau_jac + plateau_jac_size,
            0.65,
            scale=scale_jac,
        )
    else:
        solver.step_size = learning_rate.one()

    if activate_fim:
        solver.step_size_fisher = learning_rate(
            plateau_fim,
            float(jnp.log(lr)),
            plateau_fim + plateau_fim_size,
            0.65,
            # step_flat=plateau_jac + 100,
            scale=scale_fim,
        )
    else:
        solver.step_size_fisher = learning_rate.zero()

    solver.step_size_grad = learning_rate(
        plateau_grad,
        float(jnp.log(lr)),
        plateau_grad + plateau_grad_size,
        0.65,
        scale=scale_grad,
        # step_flat=100,
    )

    # solver.step_size = solver.step_size_fisher
    # solver.step_size_grad = solver.step_size_fisher
    # learning_rate.from_1_to_0(plateau_fim + 100, 0.65)
    return solver


def estim_solver(solver, niter, kwargs_run_GD, verbatim=False, **run_parameters):
    solver.verbatim = verbatim

    solver = set_solver_run_parameters(solver, **run_parameters)

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

    params_star_stack, params_star, PRNGKey = get_params_star(
        jrd.PRNGKey(int(time())), DIM_COV
    )

    data_set, sim, _ = sample(params_star, jrd.PRNGKey(0), N_IND=100, J_OBS=5)

    ls = []
    lr = []
    for i in range(1):
        params0, PRNGKey = get_random_params0(PRNGKey, params0_start)

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
        beta = theta[:, :]
        # beta_support = beta.sum(axis=0) != 0
        num_support = (beta != 0).sum(axis=0)
        print(num_support)

        id = np.array(
            [i for i in range(len(num_support)) if num_support[i] >= threshold]
        )
        xticks = [i + 1 for i in range(len(id))]
        #

        ax.boxplot(beta[:, id])
        ax.plot(xticks, params_star_stack[:][id], "bs", label="true value")
        ax.set_xticks(xticks, id)
        ax.legend()

        return fig, ax

    plot_beta(theta)
