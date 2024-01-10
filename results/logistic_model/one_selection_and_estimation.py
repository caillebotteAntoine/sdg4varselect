import jax.random as jrd
import jax.numpy as jnp

from sdg4varselect.algo import (
    BIC,
    regularization_path,
    NanError,
    variable_selection_res,
    estim_res,
)
import sdg4varselect.logistic as modelisation

from results.logistic_model.one_estim import one_estim


def _one_estim_with_selection_with_model(PRNGKey, model, dh, lbd_set):
    PRNGKey_reg_path, PRNGKey_estim = jrd.split(PRNGKey)
    # === VARIABLE SELECTION === #
    reg_path = regularization_path(
        one_estim,
        PRNGKey_reg_path,
        model,
        dh,
        lbd_set,
    )
    if reg_path is None:
        raise NanError

    DIM_LD = model.DIM_LD
    multi_theta_HD = jnp.array([res.theta[-1, DIM_LD:] for res in reg_path])
    bic = BIC(multi_theta_HD, jnp.array([res.likelihood for res in reg_path]), model.N)

    # === FINAL ESTIMATION === #
    lbd_id = jnp.argmin(bic)
    selected_component = multi_theta_HD[lbd_id] != 0
    theta_biased = reg_path[lbd_id].theta[-1]
    NEW_DIM_HD = int(selected_component.sum())

    model_shrink = modelisation.Logistic_JM(N=model.N, J=model.J, DIM_HD=NEW_DIM_HD)
    dh_shrink = dh.deepcopy()
    dh_shrink.data["cov"] = dh.data["cov"][:, selected_component]

    res_estim = one_estim(PRNGKey_estim, model_shrink, dh_shrink, lbd=None)

    # === THETA RE CONSTRUCTION === #
    id = jnp.concatenate([jnp.repeat(True, model_shrink.DIM_LD), selected_component])

    theta = jnp.zeros(shape=(model.parametrization.size,))
    theta = theta.at[jnp.where(id)].set(res_estim.theta[-1, :])

    return variable_selection_res(
        estim_res=res_estim,
        theta=theta,
        theta_biased=theta_biased,
        regularization_path=reg_path,
        bic=bic,
    )


def one_estim_with_selection(args):
    PRNGKey, N, J, DIM_HD, dh, lbd_set = args

    model = modelisation.Logistic_JM(N, J, DIM_HD)
    return _one_estim_with_selection_with_model(PRNGKey, model, dh, lbd_set)


if __name__ == "__main__":
    from sdg4varselect.logistic import sample_one, get_params_star

    lbd_set = 10 ** jnp.linspace(-2, 0, num=5)
    model = modelisation.Logistic_JM(N=100, J=5, DIM_HD=10)

    dh = sample_one(jrd.PRNGKey(0), model, weibull_censoring_loc=2000)

    selection_res = _one_estim_with_selection_with_model(
        jrd.PRNGKey(0), model, dh, lbd_set
    )

    # === PLOT === #
    reg_path = selection_res.regularization_path
    from sdg4varselect.plot import plot_theta, plot_reg_path, plot_theta_HD

    params_star = get_params_star(model.DIM_HD)

    plot_theta(reg_path, model.DIM_LD, params_star, model.params_names)
    plot_theta_HD(reg_path, model.DIM_LD, params_star, model.params_names)

    plot_reg_path(lbd_set, reg_path, selection_res.bic, model.DIM_HD)
    plot_theta(selection_res.estim_res, model.DIM_LD, params_star, model.params_names)
    plot_theta_HD(
        selection_res.estim_res, model.DIM_LD, params_star, model.params_names
    )
