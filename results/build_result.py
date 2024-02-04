import pickle
import gzip

import jax.numpy as jnp

from sdg4varselect.models.logistic_joint_model import Logistic_JM, get_params_star

from sdg4varselect.plot import (
    plot_theta,
    plot_reg_path,
    plot_theta_HD,
    plot_box_plot_HD,
    figure,
)

model = Logistic_JM(N=100, J=5, DIM_HD=100)


def read_multi_files(N, P, S=0):
    def read(n, p):
        model = Logistic_JM(N=n, J=5, DIM_HD=p)
        filename = f"test_s{S}_N{model.N}_P{model.DIM_HD}_J{model.J}"

        R = pickle.load(gzip.open(f"files/{filename}.pkl.gz", "rb"))
        print(f"{filename} LOADED")
        return R

    results = [[read(n, p) for n in N] for p in P]

    lbd_set = results[0][0]["lbd_set"]

    N = [[r["N"] for r in res] for res in results]
    P = [[r["P"] for r in res] for res in results]
    results = [[r["res"][0] for r in res] for res in results]

    if len(results[0]) == 1:
        results = [results[i][0] for i in range(len(results))]
        N = [N[i][0] for i in range(len(results))]
        P = [P[i][0] for i in range(len(results))]

    if len(results) == 1:
        results = results[0]
        N = N[0]
        P = P[0]

    return results, lbd_set, N, P


if __name__ == "__main__":
    N = (100,)
    P = (50, 100)

    N = (50, 100, 200, 300)
    P = (200,)
    scenarios_labels = (
        [f"N = {n}" for n in N] if len(P) == 1 else [f"P = {p}" for p in P]
    )

    results, lbd_set, N, P = read_multi_files(N, P)

    # ====================================================== #
    def theta_box_plot(theta, params_star, params_names, title):
        fig = figure()

        for i in range(7):
            ax = fig.add_subplot(3, 3, 1 + i)
            ax.ticklabel_format(style="sci", scilimits=(-3, 3))
            bp = ax.boxplot(theta[:, i], patch_artist=True)

            for patch in bp["boxes"]:
                patch.set(facecolor=f"C{i}")

            for median in bp["medians"]:
                median.set_color("black")

            ax.axhline(y=params_star[i], color="k", label="true value")

            ax.legend()
            ax.set_title(f"{title} {params_names[i]} ")

        return fig, fig.axes

    params_names = model.params_names
    params_star = get_params_star(P[0])
    theta = jnp.array([[r.theta for r in res] for res in results])

    title = "EMV of"

    fig = figure()

    for i in range(7):
        ax = fig.add_subplot(3, 3, 1 + i)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3))
        bp = ax.boxplot(theta[:, :, i], patch_artist=True, labels=scenarios_labels)

        for patch in bp["boxes"]:
            patch.set(facecolor=f"C{i}")

        for median in bp["medians"]:
            median.set_color("black")

        ax.axhline(y=params_star[i], color="k", label="true value")

        ax.legend()
        ax.set_title(f"{title} {params_names[i]} ")

    theta_biased = jnp.array([r.theta_biased for r in results[0]])
    theta_box_plot(theta_biased, get_params_star(P[0]), model.params_names, "biased")
    theta_box_plot(theta, get_params_star(P[0]), model.params_names, "EMV of")

    # ====================================================== #
    lbd_selected = jnp.array([[jnp.argmin(r.bic) for r in res] for res in results])

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    bp = ax.boxplot(lbd_selected, patch_artist=True)

    # ====================================================== #
    def rmse(x, x_star):
        return jnp.sqrt(((x - x_star) ** 2).mean())  # axis=1))

    def rrmse(x, x_star):
        """
        \sqrt{1/N \sum_{i=1}^n (x_i-x_{star,i})^2}"""
        return rmse(x / x_star, 1)
        return jnp.sqrt(((x / x_star - 1) ** 2).mean())
        return rmse(x, x_star) / jnp.sqrt((x**2).sum())  # axis=1))

    def errors(x, x_star, DIM_LD):
        return (
            rrmse(x[:DIM_LD], x_star[:DIM_LD]),
            rmse(x[DIM_LD:], x_star[DIM_LD:]) / jnp.abs(x_star[DIM_LD:]).max(),
        )

    res_error_rmse = jnp.array(
        [
            [
                rmse(
                    res.theta[: model.DIM_LD],
                    jnp.hstack(get_params_star(P[i]))[: model.DIM_LD],
                )
                for res in results[i]
            ]
            for i in range(len(results))
        ]
    )

    res_errors = jnp.array(
        [
            [
                errors(res.theta, jnp.hstack(get_params_star(P[i])), model.DIM_LD)
                for res in results[i]
            ]
            for i in range(len(results))
        ]
    )

    fig = figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.boxplot(res_errors[:, :, 0], labels=scenarios_labels)
    ax.set_xlabel("censorship")
    ax.set_title("rrmse of nu")

    ax = fig.add_subplot(2, 1, 2)
    ax.boxplot(res_errors[:, :, 1], labels=scenarios_labels)
    ax.set_xlabel("censorship")
    ax.set_title("rrmse of beta")

    def get_latex_table():
        out = "\\begin{tabular}{lll|}\n"
        out += "\t & \\multicolumn{2}{c}{Errors} \\\\ \\cline{2-3} \n"
        out += "Scenarios \t& $rrmse(\\beta)$\t& $rrmse(\\nu)$ \\\\ \\hline\n"

        err = res_errors.mean(axis=1)

        for i in range(err.shape[0]):
            out += f"${scenarios_labels[i]}$ \t& {err[i,1]:,.3f} \t& {err[i,0]:,.3f} \t \\\\"
            out += "\n" if i < err.shape[0] - 1 else ""

        return out + " \\hline \n\\end{tabular}"

    print(get_latex_table())
