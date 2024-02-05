import jax.numpy as jnp

from sdg4varselect.models.logistic_joint_model import Logistic_JM, get_params_star

import sdg4varselect.plot as sdgplt
from sdg4varselect.outputs import sdg4vsResults


# results, lbd_set, N, P = read_multi_files(N=(50, 100, 200, 300), P=(200,))


# # === PLOT === #
# res = results[0]

# selection_res = res[0][0]
# reg_path = selection_res.regularization_path


# model = Logistic_JM(N=20, J=5, DIM_HD=200)
# params_star = get_params_star(model.DIM_HD)
# params_names = model.params_names

# plot_theta(reg_path, model.DIM_LD, params_star, params_names)
# plot_reg_path(lbd_set, reg_path, selection_res.bic, model.DIM_HD)

# estim_res = [r.estim_res for r in res]
# theta = jnp.array([r.theta for r in res])
# theta_biased = jnp.array([r.theta_biased for r in res])

# plot_theta(estim_res, model.DIM_LD, params_star, params_names)
# plot_theta_HD(estim_res, model.DIM_LD, params_star, params_names)
# plot_box_plot_HD(theta, model.DIM_LD, params_star)


# # ====================================================== #
