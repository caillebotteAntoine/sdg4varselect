# Create by antoine.caillebotte@inrae.fr

import numpy as np
import pickle

import sdgplt

folder = "images"


def get_data(filename):
    print(f"====== open file {filename} ======")
    with open(f"../run_script/results/{filename}.pkl", "rb") as f:
        data = pickle.load(f)
    # 50 full random beta0
    # with open("../run_script/1699274542_multi_100_50_30_5_0.pkl", "rb") as f:
    #     data = pickle.load(f)

    # 200 + censure 20%
    # with open("../run_script/1699246117_multi_100_200_5_20.pkl", "rb") as f:
    #     data = pickle.load(f)

    params_star_stack = data["params_star_stack"]
    params_names = data["params_names"]
    lbd_set = data["lbd_set"]
    theta = np.array(data["theta"])
    n_run_base = theta.shape[0]
    theta_biais = np.array(data["theta_biais"])
    bic = data["lbic"]

    id = [
        [i for i in range(len(theta)) if np.isnan(theta[i]).any()],
        [i for i in range(len(bic)) if np.isinf(bic[i]).any()],
        [i for i in range(len(theta)) if theta[i, 7:].mean() == 0],
        # [i for i in range(len(theta)) if np.abs(theta[i, 1] - 90) > 10],
    ]

    id = np.unique(np.concatenate(id))
    id = [i for i in range(len(theta)) if i not in id]

    bic = [data["lbic"][i] for i in id]
    ebic = [data["lebic"][i] for i in id]
    theta_reg = [data["ltheta_reg"][i] for i in id]

    theta = theta[id]
    theta_biais = theta_biais[id]

    n_run = theta.shape[0]
    print(f"p = {theta.shape[1]-7}, nrun = {n_run} (remove {n_run_base-n_run})")
    return (
        params_star_stack,
        params_names,
        lbd_set,
        bic,
        theta_reg,
        ebic,
        theta,
        theta_biais,
    )


# from sdg4varselect import jrd
# from joint_model.sample import get_params_star

# params_star_stack, _, prng_key = get_params_star(jrd.PRNGKey(0), theta.shape[1] - 7)


# ====================================================== #
def rmse(x, x_star):
    return np.sqrt(((x - x_star) ** 2).mean(axis=1))


def rrmse(x, x_star):
    return rmse(x, x_star) / np.sqrt((x**2).sum(axis=1))


filenames = ["0", "19", "41", "60", "79"]
main = "1702481006_multi_N100_P100_J5_C"

filenames = ["0", "19", "40", "59", "80"]
main = "1702551174_multi_N100_P200_J5_C"

lrmse_beta = []
lrmse_nu = []
for f in filenames:
    (
        params_star_stack,
        params_names,
        lbd_set,
        bic,
        theta_reg,
        ebic,
        theta,
        theta_biais,
    ) = get_data(f"{main}{f}")

    lrmse_beta.append(rrmse(theta[:, 7:], params_star_stack[7:]))
    lrmse_nu.append(rrmse(theta[:, :7], params_star_stack[:7]))

# lrmse_beta = np.array(lrmse_beta)
# lrmse_nu = np.array(lrmse_nu)


fig = sdgplt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.boxplot(lrmse_beta, labels=[f"{c}%" for c in filenames])
ax.axhline(y=0.05, color="k")
ax.set_xlabel("censorship")
ax.set_title("rrmse of beta")

ax = fig.add_subplot(2, 1, 2)
ax.boxplot(lrmse_nu, labels=[f"{c}%" for c in filenames])
ax.axhline(y=0.05, color="k")
ax.set_xlabel("censorship")
ax.set_title("rrmse of nu")

# ====================================================== #
(
    params_star_stack,
    params_names,
    lbd_set,
    bic,
    theta_reg,
    ebic,
    theta,
    theta_biais,
) = get_data(f"{main}{filenames[0]}")

id = np.abs(theta[:, 1] - 90) < 10
theta[id, 1]

# ====================================================== #

for i in range(len(bic) // 10):
    fig, axs = sdgplt.plot_regularization_path(theta_reg[i], lbd_set, bic[i])
    ax, ax_bic = axs
    ax.set_title(f"{i}")


# ====================================================== #
def boxplot(t):
    fig = sdgplt.figure()

    for i in range(7):
        ax = fig.add_subplot(3, 3, 1 + i)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3))
        bp = ax.boxplot(t[:, i], patch_artist=True)

        for patch in bp["boxes"]:
            patch.set(facecolor=f"C{i}")

        for median in bp["medians"]:
            median.set_color("black")

        ax.axhline(y=params_star_stack[i], color="k", label="true value")

        ax.legend()
    return fig, ax

    fig, ax = boxplot(theta_biais)
    ax.set_title(f"{params_names[i]} biased")


# ====================================================== #

fig, ax = boxplot(theta)
ax.set_title(f"EMV of {params_names[i]}")

# ====================================================== #
fig = sdgplt.figure()

for i in range(7):
    ax = fig.add_subplot(3, 3, 1 + i)
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    bp = ax.boxplot(
        np.abs(theta[:, i] / params_star_stack[i] - 1),
        patch_artist=True,
        showfliers=False,
    )

    for patch in bp["boxes"]:
        patch.set(facecolor=f"C{i}")

    for median in bp["medians"]:
        median.set_color("black")

    ax.axhline(y=5 / 100, color="k", label="5% error")

    ax.legend()

    ax.set_title(f"relative error of {params_names[i]}")


# ====================================================== #
def plot_beta(theta, threshold=0):
    fig = sdgplt.figure()
    ax = fig.add_subplot(1, 1, 1)
    beta = theta[:, 7:]
    # beta_support = beta.sum(axis=0) != 0
    num_support = (beta != 0).sum(axis=0)
    print(num_support)

    id = np.array([i for i in range(len(num_support)) if num_support[i] >= threshold])
    xticks = [i + 1 for i in range(len(id))]
    #

    ax.boxplot(beta[:, id])
    ax.plot(xticks, params_star_stack[7:][id], "bs", label="true value")
    ax.set_xticks(xticks, id)
    ax.legend()

    return fig, ax


fig, ax = plot_beta(theta_biais, theta.shape[0] / 10)
ax.set_title("biaised beta")

fig, ax = plot_beta(theta, theta.shape[0] / 10)
ax.set_title("EMV beta")
# ====================================================== #
