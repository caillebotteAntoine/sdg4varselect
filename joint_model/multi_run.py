# Create by antoine.caillebotte@inrae.fr

from one_run import kwargs_run_GD, sample_and_estim, params0, N_IND, DIM_COV
from sdg4varselect.miscellaneous import step_message

from sdg4varselect import jrd, jnp
import sdg4varselect.plot as sdgplt


def regularization_path(path, prng_key, nrep=1, verbatim=False):
    res_solver = []

    for i in range(len(path)):
        print(step_message(i, len(path)), end="\r" if not verbatim else "\n")

        res_solver.append([])

        kwargs_run_GD["prox_regul"] = path[i]
        kwargs_run_GD["proximal_operator"] = True
        for _ in range(nrep):
            _, solver, prng_key = sample_and_estim(params0, prng_key, verbatim=verbatim)

            res_solver[-1].append(solver)

    return res_solver, prng_key


# ====================================================== #
# ================ REGULARIZATION PATH ================= #
# ====================================================== #
lbd_set = 10 ** jnp.linspace(-1.5, 1, num=5)

res_solver = regularization_path(lbd_set, jrd.PRNGKey(0), verbatim=False)

fig, ax, bic_res = sdgplt.plot_regularization_path(
    res_solver,
    lbd_set,
    p=DIM_COV,
    N=N_IND,
)

ax[0].title.set_fontsize(28)
ax[0].xaxis.label.set_fontsize(28)
ax[0].yaxis.label.set_fontsize(28)
fig.savefig("images/regularization_path.png")
