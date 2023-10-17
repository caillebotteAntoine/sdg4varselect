# Create by antoine.caillebotte@inrae.fr

import numpy as np
import pickle

import sdg4varselect.plot as sdgplt

from one_run import (
    get_random_params0,
    params_star_weibull,
    params_star_stack,
    N_IND,
    DIM_COV,
    get_solver,
)

folder = "images"

# 200_50_simple_grad_10_rep
with open("res_multi_run.pkl", "rb") as f:
    data = pickle.load(f)

params_names = data["params_names"]
theta = np.array(data["theta"])


params_names = ["mu1", "mu2", "mu3", "gamma2_1", "gamma2_2", "sigma2", "alpha", "beta"]

theta = np.array(data["theta"])
fig = sdgplt.figure()

for i in range(7):
    ax = fig.add_subplot(3, 3, 1 + i)
    ax.ticklabel_format(style="sci", scilimits=(-3, 3))
    bp = ax.boxplot(theta[:, i], patch_artist=True)

    for patch in bp["boxes"]:
        patch.set(facecolor=f"C{i}")

    ax.axhline(y=params_star_stack[i], color="k", label="true value")

    ax.legend()
    ax.set_title(params_names[i])

beta = theta[:, 7:]
beta_support = beta.sum(axis=0) != 0
id = [i for i in range(len(beta_support)) if beta_support[i]]
print(beta[:, id])

sdgplt.plt.boxplot(beta[:, id])
