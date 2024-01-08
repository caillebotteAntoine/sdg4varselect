import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.algo import BIC, SPG_FIM, regularization_path, NanError
import sdg4varselect.pharmacokinetic as modelisation

from one_estim import one_estim, algo_settings


def _one_estim_with_selection_with_model(PRNGKey, model, dh, algo_settings, lbd_set):
    PRNGKey_reg_path, PRNGKey_estim = jrd.split(PRNGKey)
    # === VARIABLE SELECTION === #
    reg_path = regularization_path(
        one_estim,
        PRNGKey_reg_path,
        model,
        dh,
        algo_settings,
        lbd_set,
    )
    if reg_path is None:
        raise NanError

    multi_theta = jnp.array([res.theta for res in reg_path])
    DIM_LD = model.DIM_LD
    multi_theta_HD = multi_theta[:, -1, DIM_LD:]
    bic = BIC(multi_theta_HD, jnp.array([res.likelihood for res in reg_path]), model.N)

    # === FINAL ESTIMATION === #
    lbd_id = jnp.argmin(bic)
    selected_component = multi_theta_HD[lbd_id] != 0
    NEW_DIM_HD = int(selected_component.sum())

    model_shrink = modelisation.pharma_JM(N=model.N, J=model.J, DIM_HD=NEW_DIM_HD)
    dh_shrink = dh.deepcopy()
    dh_shrink.data["cov"] = dh.data["cov"][:, selected_component]

    res_estim = one_estim(
        PRNGKey_estim, model_shrink, dh_shrink, algo_settings, lbd=None
    )

    # === THETA RE CONSTRUCTION === #
    id = jnp.concatenate([jnp.repeat(True, model_shrink.DIM_LD), selected_component])

    theta = jnp.zeros(shape=(model.parametrization.size,))
    theta.at[jnp.where(id)].set(res_estim.theta[-1, :])

    return SPG_FIM.variable_selection_res(
        estim_res=res_estim, theta=theta, regularization_path=reg_path, bic=bic
    )


def one_estim_with_selection(args):
    PRNGKey, N, J, DIM_HD, dh, lbd_set = args

    model = modelisation.pharma_JM(N, J, DIM_HD)
    return _one_estim_with_selection_with_model(
        PRNGKey, model, dh, algo_settings, lbd_set
    )


if __name__ == "__main__":
    lbd_set = 10 ** jnp.linspace(-2, 1, num=10)
    model = modelisation.pharma_JM(N=100, J=12, DIM_HD=10)

    dh = modelisation.sample_one(jrd.PRNGKey(1), model, weibull_censoring_loc=2000)

    # selection_res = _one_estim_with_selection_with_model(
    #     jrd.PRNGKey(0), model, dh, algo_settings, lbd_set
    # )

    # # === PLOT === #
    # reg_path = selection_res.regularization_path

    reg_path = regularization_path(
        one_estim,
        jrd.PRNGKey(1),
        model,
        dh,
        algo_settings,
        lbd_set,
    )
    print([res.theta.shape for res in reg_path])
    # from sdg4varselect.plot import plot_theta, plot_reg_path, plot_theta_HD

    # params_star = modelisation.get_params_star(model.DIM_HD)

    # plot_theta(reg_path, model.DIM_LD, params_star, model.params_names)
    # plot_reg_path(lbd_set, reg_path, selection_res.bic, model.DIM_HD)
    # plot_theta(selection_res.estim_res, model.DIM_LD, params_star, model.params_names)
    # plot_theta_HD(
    #     selection_res.estim_res, model.DIM_LD, params_star, model.params_names
    # )
