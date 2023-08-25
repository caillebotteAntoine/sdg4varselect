from sdg4varselect import jnp, sdgplt

solver_selection = 0
res_selection = 0
params_star_weibull = 0


jnp.linalg.eigvalsh(res_selection.fisher_info[532])


_ = sdgplt.plt.plot(
    [
        jnp.linalg.eigvalsh(res_selection.fisher_info[i])
        for i in range(len(res_selection.fisher_info))
    ]
)


solver_selection.likelihood_kwargs["Y"]


res_selection.fisher_info[500]


def get_cov_product(solver, param_star):
    cov = solver.likelihood_kwargs["cov"]

    print(f"beta^T cov = {cov @ solver.params.beta}")
    print(f"beta_*^T cov = {cov @ param_star.beta}")


get_cov_product(solver_selection, params_star_weibull)

_ = sdgplt.plt.plot(res_selection.grad[490:510, :])

solver_selection.step_size.plot()


_ = sdgplt.plt.plot([res_selection.jac[i].max() for i in range(len(res_selection.jac))])


_ = sdgplt.plt.plot(
    [res_selection.fisher_info[i].max() for i in range(len(res_selection.fisher_info))]
)


_ = sdgplt.plt.plot(res_selection.grad_precond[490:, :])


jac = jac_likelihood(res_selection.theta[-2], **solver_selection.likelihood_kwargs)

grad = jac.mean(axis=0)
fim = jac.T @ jac

grad_precond = jnp.linalg.solve(fim, grad)


res_selection.jac[-1]

for i in range(1, 5):
    jac = res_selection.jac[-i]

    grad = jac.mean(axis=0)
    print(f"{grad.max()}  et {res_selection.grad[-i].max()}")

    fim = jac.T @ jac

    grad_precond = jnp.linalg.solve(fim, grad)
    print(f"{grad_precond.max()}  et {res_selection.grad_precond[-i].max()}\n")
