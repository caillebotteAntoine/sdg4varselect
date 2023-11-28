# Create by antoine.caillebotte@inrae.fr
from time import time
from datetime import datetime
from sdg4varselect.miscellaneous import time2string, list_to_BIC, step_message

import numpy as np

from sdg4varselect.solver import shrink_support

from joint_model.one_run import estim, estim_solver

from sdg4varselect import jnp


kwargs_run_GD = {
    "prox_regul": 1.29e-3,
    "proximal_operator": False,
}


def one_res(
    parameters0,
    data_set,
    lbd,
    prng_key,
    verbatim=False,
):
    kwargs_run_GD["prox_regul"] = lbd
    kwargs_run_GD["proximal_operator"] = True

    res, solver, error_flag, key = estim(
        data_set,
        parameters0,
        prng_key,
        niter=2000,
        kwargs_run_GD=kwargs_run_GD,
        verbatim=verbatim,
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

    return solver, res, error_flag, key


def regularization_path(
    parameters0,
    data_set,
    path,
    DIM_COV,
    prng_key,
    verbatim=False,
):
    list_res = []
    list_solver = []

    for i in range(len(path)):
        print(step_message(i, len(path)), end="\r" if not verbatim else "\n")
        solver, res, error_flag, key = one_res(
            parameters0, data_set, path[i], prng_key, verbatim
        )

        if error_flag:
            print("error detected cancel regularization path !")
            return -1

        list_solver.append(solver)
        list_res.append(res)

        # print(f"#beta = {solver.get_number_of_nonzero(p=DIM_COV)}")

        if solver.get_number_of_nonzero(p=DIM_COV) == 0:
            for k in range(len(path) - i - 1):
                # print(
                #     step_message(i + k, len(path)), end="\r" if not verbatim else "\n"
                # )
                list_solver.append(list_solver[-1])
                list_res.append(list_res[-1])

            if verbatim:
                print(f"break at {path[i]}")
            break

    print(step_message(len(path), len(path)), end="\n")
    return list_solver, list_res, key


def final_estim(solver, parameters0, prox_regul, verbatim=False):
    kwargs_run_GD["prox_regul"] = prox_regul
    kwargs_run_GD["proximal_operator"] = False
    p0 = parameters0.copy()
    (DIM_COV,) = p0["beta"].shape

    solver_select, mask_select = shrink_support(solver, "beta", DIM_COV)
    p0["beta"] = jnp.where(mask_select[-DIM_COV:], p0["beta"], 0)
    solver_select.reset_solver()
    solver_select.theta_reals1d = p0

    res, solver_select, error_flag = estim_solver(
        solver_select,
        niter=2000,
        kwargs_run_GD=kwargs_run_GD,
        verbatim=verbatim,
        activate_fim=True,
        activate_jac_approx=True,
        lr=1e-8,
        # Grad
        plateau_grad=1000,
        plateau_grad_size=100,
        scale_grad=1,
        # Jac
        plateau_jac=1000,
        plateau_jac_size=2000,
        scale_jac=1,
        # Fim
        plateau_fim=1000,
        plateau_fim_size=1000,
        scale_fim=0.9,
    )

    if error_flag:
        print("error detected cancel final estimation !")
        return -1

    return res, solver_select


def method(
    params0,
    data_set,
    path,
    N_IND,
    DIM_COV,
    prng_key,
    verbatim=True,
):
    # ====================================================== #
    # ================ REGULARIZATION PATH ================= #
    # ====================================================== #
    time_start = time()
    res = regularization_path(
        params0,
        data_set,
        path,
        DIM_COV,
        prng_key=prng_key,
        verbatim=verbatim,
    )
    print(f"REGULARIZATION PATH TIME: {time2string(time() - time_start)}")

    if res != -1:
        ls, lr, prng_key = res
    else:
        return -1

    bic, ebic, theta_reg = list_to_BIC(ls, lr, N_IND, DIM_COV, verbatim=False)

    # ============================================ #
    # ================ INFERENCE ================= #
    # ============================================ #
    bic_argmin = np.argmin(bic)
    res_selection = lr[bic_argmin]
    solver_selection = ls[bic_argmin]
    lbd_selection = path[bic_argmin]

    res = final_estim(solver_selection, params0, lbd_selection, verbatim=verbatim)

    if res != -1:
        final_res = res[0]
        final_solver = res[1]
    else:
        return -1

    return (
        final_res,
        final_solver,
        solver_selection,
        res_selection,
        bic,
        ebic,
        theta_reg,
        lbd_selection,
        ls,
        lr,
    )


if __name__ == "__main__":
    # from joint_model.sample import get_params_star

    pass
    # from work import sdgplt

    # # ====================================================== #
    # print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    # seed = 1697807204  #
    # seed = int(time())  #
    # print(f"seed = {seed}")
    # prng_key = jrd.PRNGKey(seed)
    # # ====================================================== #

    # DIM_COV = 10
    # N_IND = 20
    # J_OBS = 5
    # CENSORING = 0

    # params_star_stack, params_star_weibull = get_params_star(DIM_COV)

    # params0 = {
    #     "mu1": 0.5,  # 1
    #     "mu2": 50.0,  # 2
    #     "mu3": 3.0,  # 3
    #     "gamma2_1": 0.00025,  # 4
    #     "gamma2_2": 2.0,  # 5
    #     "sigma2": 0.0001,  # 6
    #     "alpha": 5.0,  # 7
    #     "beta": np.random.uniform(-1, 1, size=DIM_COV),
    # }
    # lbd_set = 10 ** jnp.linspace(-2, 0, num=10)
    # # lbd_set = [0.25]
    # # ====================================================== #

    # res = method(
    #     params0,
    #     params_star_weibull,
    #     lbd_set,
    #     N_IND,
    #     DIM_COV,
    #     J_OBS,
    #     CENSORING,
    #     prng_key=prng_key,
    #     verbatim=True,
    # )

    # if res != -1:
    #     (
    #         final_res,
    #         final_solver,
    #         solver_selection,
    #         res_selection,
    #         bic,
    #         ebic,
    #         theta_reg,
    #         lbd_selection,
    #         ls,
    #         lr,
    #     ) = res

    #     # ====================================================== #
    #     # fig, axs = sdgplt.plot_regularization_path(theta_reg, lbd_set, bic)

    #     # ax, ax_bic = axs
    #     # ax_ebic = ax.twinx()
    #     # ax_ebic.plot(lbd_set, ebic, color="r", linewidth=2, linestyle="--", label="eBIC")
    #     # id_min = np.nanargmin(ebic)
    #     # sdgplt.plot_axvline(ax_ebic, lbd_set, ebic, id_min, color="g", msg="min(eBIC)")
    #     # ax_ebic.legend(loc="upper right")

    #     # for res in lr[0]:
    #     #     _, _ = sdgplt.plot_params(
    #     #         x=res.theta,
    #     #         x_star=np.array(params_star_stack),
    #     #         p=DIM_COV,
    #     #         names=final_solver.params_names,
    #     #         logscale=False,
    #     #     )

    #     # ====================================================== #
    #     params_names = solver_selection.params_names

    #     fig = sdgplt.figure()
    #     solver_selection.step_size.plot(label="Jac step size")
    #     solver_selection.step_size_fisher.plot(label="FIM step size")
    #     solver_selection.step_size_grad.plot(label="gradient step size")
    #     sdgplt.plt.legend()

    #     _, _ = sdgplt.plot_params_grad(
    #         res_selection.theta,
    #         res_selection.grad_precond,
    #         np.array(params_star_stack),
    #         p=DIM_COV,
    #         names=params_names,
    #         logscale=False,
    #     )

    #     _, _ = sdgplt.plot_params_hd(res_selection.theta, p=DIM_COV, location="right")

    #     print(res_selection.theta[-1][:DIM_COV])

    #     # =========================#
    #     _, _ = sdgplt.plot_params_grad(
    #         final_res.theta,
    #         final_res.grad_precond,
    #         np.array(params_star_stack),
    #         p=DIM_COV,
    #         names=params_names,
    #         logscale=False,
    #     )

    #     _, _ = sdgplt.plot_params_hd(final_res.theta, p=DIM_COV, location="right")

    #     print(final_res.theta[-1][:DIM_COV])

    #     print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

    # ====================================================== #
    # def extract_data(res, solver):
    #     latent_variables = ls[np.argmin(bic)].latent_variables
    #     for var in latent_variables.values():
    #         var.likelihood = None

    #     data = {
    #         "theta": res.theta,
    #         "grad_precond": res.grad_precond,
    #         "likelihood": res.likelihood,
    #         "latent_variables": latent_variables,
    #         "jac_min": [res.jac[i].min() for i in range(len(res.jac))],
    #         "jac_max": [res.jac[i].max() for i in range(len(res.jac))],
    #         "fim_det": [jnp.linalg.det(x) for x in res.fisher_info],
    #         "fim_vp": np.array([jnp.linalg.eigvalsh(x) for x in res.fisher_info]),
    #     }
    #     return data

    # data_selection = extract_data(res_selection, ls[np.argmin(bic)])

    # data_final = extract_data(final_res, final_solver) if final_res != -1 else -1

    # step_size = {
    #     "jac": final_solver.step_size,
    #     "fisher": final_solver.step_size_fisher,
    #     "gradient": final_solver.step_size_grad,
    # }

    # data = {
    #     "res_selection": data_selection,
    #     "res_final": data_final,
    #     "bic": bic,
    #     "ebic": ebic,
    #     "theta_reg": theta_reg,
    #     "lbd_set": lbd_set,
    #     "params_names": ls[0].params_names,
    #     "step_size": step_size,
    #     "DIM_COV": DIM_COV,
    #     "N_IND": N_IND,
    #     "params_star_stack": params_star_stack,
    # }

    # with open("res_selection.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # print("RESULT SAVED !")
