import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import jax.numpy as jnp

from functools import wraps


def dec_figsize(func):
    argcount = func.__code__.co_argcount - 1
    if func.__code__.co_varnames[argcount] != "figsize":
        raise ValueError("the last argument must be figsize")

    if len(func.__defaults__) == 0:
        raise ValueError("figsize must have a default value")
    figsize = func.__defaults__[-1]

    @wraps(func)
    def new_func(*args, **kwargs):
        out = func(*args, **kwargs)
        if out is not None and len(out) > 0:
            out[0].set_figheight(figsize)
            out[0].set_figwidth(figsize)

        return out

    return new_func


def remove_nan(x):
    return x[~jnp.isnan(x).any(axis=1)]


def plot_mcmc(x):
    """plot an MCMC_chain"""
    import matplotlib.pyplot as plt
    from sdg4varselect.MCMC import MCMC_chain
    from sdg4varselect.solver import solver

    if isinstance(x, solver):
        return [plot_mcmc(mcmc) for mcmc in x.latent_variables.values()]

    if isinstance(x, MCMC_chain):
        if len(x.sd) == 1:
            fig, axs = plt.subplots(2, 1, sharex=True)
        else:
            fig, axs = plt.subplots(3, 1, sharex=True)

        axs[0].set_title(label="chaine de " + x.name)

        axs[0].plot(x.chain)
        axs[0].set_ylabel("chain")

        axs[1].plot(x.acceptance_rate())
        axs[1].set_ylabel("acceptance_rate")

        if len(x.sd) != 1:
            axs[2].plot(x.sd)
            axs[2].set_ylabel("proposal sd")

        axs[-1].set_xlabel("Iteration")

        return fig, axs


def ax_plot_line_with_doted_hline(
    fig, nrow, ncol, i, x, x_star=None, color=None, label=None, logscale=True
):
    ax = fig.add_subplot(nrow, ncol, i)

    if x_star is not None:
        if color is None:
            ax.hlines(
                y=x_star,
                xmin=0,
                xmax=len(x),
                linestyles="--",
            )
        else:
            ax.hlines(
                y=x_star,
                xmin=0,
                xmax=len(x),
                colors=color,
                linestyles="--",
            )

    if label is None:
        if color is None:
            ax.plot(x)
        else:
            ax.plot(x, color=color)
    else:
        if color is None:
            ax.plot(x, label=label)
        else:
            ax.plot(x, color=color, label=label)
        ax.legend(loc="center left")

    if logscale:
        ax.set_yscale("log")

    return fig, ax


def ax_plot_multi_line_with_doted_hline(
    fig, nrow, ncol, i, x, x_star=None, p=None, label=None, logscale=True, title=None
):
    if not isinstance(i, tuple):
        return ax_plot_multi_line_with_doted_hline(
            fig, nrow, ncol, (i, i), x, x_star, p, label, logscale, title
        )

    x = remove_nan(x)

    dim = x.shape[1]
    if x_star is not None:
        star = jnp.hstack(x_star)
        assert len(star) == dim

    if label is not None:
        assert len(label) == dim

    if p is not None:
        dim -= p

    ax = []
    for j in range(dim):
        i_new = tuple([k + j * ncol for k in i])

        fig, ax0 = ax_plot_line_with_doted_hline(
            fig,
            nrow,
            ncol,
            i_new,
            x=x[:, j],
            x_star=None if x_star is None else star[j],
            color=f"C{j}",
            label=None if label is None else label[j],
            logscale=logscale,
        )
        ax.append(ax0)

    if title is not None:
        if len(ax) != 0:
            ax[0].set_title(title)

    return fig, ax


@dec_figsize
def plot_params(x, x_star=None, p=None, names=None, logscale=True, figsize=15):
    fig, ax = ax_plot_multi_line_with_doted_hline(
        plt.figure(),
        x.shape[1] - p,
        1,
        1,
        x,
        x_star,
        p,
        label=names,
        logscale=logscale,
        title="Parameter",
    )

    ax[-1].set_xlabel("Iteration")
    return fig, ax


@dec_figsize
def plot_grad(x, p=None, names=None, figsize=15):
    fig, ax = ax_plot_multi_line_with_doted_hline(
        plt.figure(),
        x.shape[1] - p,
        1,
        1,
        x,
        x_star=np.zeros(shape=x[0].shape),
        p=p,
        label=names,
        logscale=False,
        title="Gradient",
    )

    ax[-1].set_xlabel("Iteration")
    return fig, ax


@dec_figsize
def plot_params_grad(
    params, grad, params_star=None, p=None, names=None, logscale=True, figsize=15
):
    dim = params.shape[1] - p
    fig, ax0 = ax_plot_multi_line_with_doted_hline(
        plt.figure(),
        dim,
        2,
        1,
        x=params,
        x_star=params_star,
        p=p,
        label=names,
        logscale=logscale,
        title="Parameter",
    )

    fig, ax1 = ax_plot_multi_line_with_doted_hline(
        fig,
        dim,
        2,
        2,
        x=grad,
        x_star=np.zeros(shape=grad[0].shape),
        p=p,
        label=None,
        logscale=False,
        title="Gradient",
    )

    ax = ax0 + ax1
    return fig, ax


def ax_plot_list_of_vector(
    fig,
    nrow,
    ncol,
    i,
    x,
    p,
    x_axs=None,
    colormap="RdBu_r",
    logscale=False,
    title=None,
    colorbar=True,
    location="bottom",
):
    x = remove_nan(x)
    x_hd = x[:, -p:].T

    x_abs = x_axs if x_axs is not None else 0.5 + np.arange(0, x_hd.shape[1] + 1, 1)

    y_ord = 0.5 + np.arange(0, p + 1, 1)

    ax = fig.add_subplot(nrow, ncol, i)

    vmin = x_hd.min() if x_hd.min() < 0 else -0.001
    vmax = x_hd.max() if x_hd.max() > 0 else 0.0001

    colormesh = ax.pcolormesh(
        x_abs,
        y_ord,
        x_hd,
        cmap=plt.colormaps[colormap] if isinstance(colormap, str) else colormap,
        norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
    )
    if colorbar:
        fig.colorbar(colormesh, ax=ax, location=location)

    if logscale:
        ax.set_xscale("log")

    if title is not None:
        ax.set_title(title)

    return fig, ax


@dec_figsize
def plot_params_hd(x, p, colormap="RdBu_r", location="bottom", figsize=15):
    return ax_plot_list_of_vector(
        plt.figure(),
        1,
        1,
        1,
        x,
        p,
        x_axs=None,
        colormap=colormap,
        logscale=False,
        title="HD parameter",
        location=location,
    )


@dec_figsize
def plot_grad_hd(x, p, colormap="RdBu_r", figsize=15):
    return ax_plot_list_of_vector(
        plt.figure(),
        1,
        1,
        1,
        x,
        p,
        x_axs=None,
        colormap=colormap,
        logscale=False,
        title="HD gradient",
    )


@dec_figsize
def plot_params_grad_hd(params, grad, p, mask=None, colormap="RdBu_r", figsize=15):
    fig = plt.figure()

    fig, ax0 = ax_plot_list_of_vector(
        fig,
        1,
        16,
        (1, 7 if mask is None else 6),
        x=params,
        p=p,
        x_axs=None,
        colormap=colormap,
        logscale=False,
        title="HD parameter",
    )

    if mask is not None:
        x = -1 + 2 * jnp.array([mask])

        colors.LinearSegmentedColormap.from_list
        cmap = colors.ListedColormap(["red", "violet", "blue"])
        fig, ax_mask = ax_plot_list_of_vector(
            fig, 1, 16, 7, x, p, colormap=cmap, colorbar=True
        )

        ax_mask.axis("off")

    fig, ax1 = ax_plot_list_of_vector(
        fig,
        1,
        16,
        (9, 16),
        x=grad,
        p=p,
        x_axs=None,
        colormap=colormap,
        logscale=False,
        title="HD gradient",
    )

    return fig, [ax0, ax1]


def plot_all_params_grad(
    params,
    grad,
    params_star=None,
    p=None,
    mask=None,
    names=None,
    logscale=True,
    figsize=15,
):
    fig0, ax0 = plot_params_grad(
        params, grad, params_star, p, names, logscale, figsize=figsize
    )

    fig1, ax1 = plot_params_grad_hd(params, grad, p, mask=mask, figsize=figsize)

    return [fig0, fig1], [ax0, ax1]


def get_theta_HD_and_BIC_value(res_solver, p, N=None):
    theta_regularization = [
        np.array([x.theta_reals1d for x in y]).mean(axis=0)[-p:] for y in res_solver
    ]

    if N is not None:
        bic = np.array([[x.BIC(N, p, size=1000) for x in y] for y in res_solver])
        return theta_regularization, bic
    else:
        return theta_regularization, None


@dec_figsize
def plot_selected_component(res_solver, lbd_set, p, colormap="bwr", figsize=10):
    theta_regularization, _ = get_theta_HD_and_BIC_value(res_solver, p=p)

    fig, ax = ax_plot_list_of_vector(
        plt.figure(),
        1,
        1,
        1,
        x=np.array(theta_regularization),
        p=p,
        x_axs=lbd_set,
        colormap=colormap,
        logscale=False,
        title="Selected components",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Regularization penalty")
    ax.set_ylabel("Selected components")

    return fig, ax


@dec_figsize
def plot_regularization_path(res_solver, lbd_set, p, N, se_percentage=None, figsize=10):
    theta_regularization, bic = get_theta_HD_and_BIC_value(res_solver, p, N)

    fig, ax = plt.subplots()
    ax.set_title("Regularization path")
    ax.set_xlabel(r"Regularization penalty (\lambda)")
    ax.set_ylabel(r"HD Parameter (\beta)")
    ax.set_xscale("log")
    ax.plot(lbd_set, theta_regularization)

    # BIC PLOT
    ax_bic = ax.twinx()
    bic_mean = bic.mean(axis=1)
    ax_bic.plot(lbd_set, bic_mean, color="k", linewidth=2, linestyle="--", label="BIC")

    bic_res = {
        "bic": bic_mean,
    }

    def plot_axvline(id, color, msg=""):
        lbd = lbd_set[id]

        ax_bic.axvline(
            x=lbd,
            color=color,
            linewidth=2,
            linestyle="--",
            label="$\lambda$" + msg,
        )
        ax_bic.text(
            lbd,
            0.8 * bic_mean.max() + 0.2 * bic_mean.min(),
            f"$\lambda$ = {lbd_set[id]:.3e}",
            ha="center",
            va="center",
            rotation="vertical",
            backgroundcolor="white",
        )

    # minimum value of bic
    id_min = np.argmin(bic_mean)
    plot_axvline(id_min, color="b", msg="min")
    bic_res["min"] = bic_mean[id_min]

    # bic_sub = bic_mean[1:] - bic_mean[:-1]
    # id_coude = np.argmax(bic_sub > 0)
    # plot_axvline(id_coude, color="g", msg=" coude")
    # bic_res["coude"] = bic_mean[id_coude]

    # One standard error max(min)
    id_1se = (
        None
        if se_percentage is None
        else np.argmax((bic_mean - id_min) < se_percentage * id_min)
    )
    bic_1se = None if id_1se is None else bic_mean[id_1se]
    if bic_1se is not None:
        plot_axvline(id_1se, color="r", msg=" 1 s.e.")
        bic_res["1se"] = bic_1se

    ax_bic.legend(loc="lower right")
    return (fig, [ax, ax_bic], bic_res)


def get_rmse(res_solver, params_star, exclude):
    names = res_solver[0][0].params_names
    names_shrink = [k for k in names if k not in exclude]

    rmse = np.array(
        [
            np.abs(
                [
                    [
                        1 - getattr(params_star, k) / getattr(x.params, k)
                        for k in names_shrink
                    ]
                    for x in y
                ]
            ).mean(axis=0)
            for y in res_solver
        ]
    )
    return rmse


@dec_figsize
def plot_rmse(res_solver, params_star, lbd_set, exclude, logscale=True, figsize=10):
    rmse = get_rmse(res_solver, params_star, exclude)
    if rmse.shape[1] != 0:
        names = res_solver[0][0].params_names
        names_shrink = [k for k in names if k not in exclude]

        fig, ax = plt.subplots()
        ax.plot(lbd_set, rmse, label=names_shrink)

        if logscale:
            ax.set_yscale("log")

        ax.set_xscale("log")
        ax.set_xlabel("Regularization penalty")
        ax.set_ylabel("Relative error for each parameter")
        ax.legend(loc="center left")

        return fig, ax
