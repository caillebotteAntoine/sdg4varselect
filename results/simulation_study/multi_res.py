"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116, W0211

from sdg4varselect import regularization_path, lasso_into_estim
from sdg4varselect.outputs import RegularizationPathRes, MultiRunRes


def add_flag(fct):
    def estim_with_flag(model, **kwargs) -> tuple[MultiRunRes, bool]:
        """must return the estimation results and
        a flag which indicates if the regularization path is finished"""
        res_estim = lasso_into_estim(fct, model=model, **kwargs)
        dim_ld = model.DIM_LD
        flag = (res_estim[-1].last_theta[dim_ld:] != 0).sum() == 0

        return res_estim, flag

    return estim_with_flag


def one_result(estim_fct_with_flag, prngkey, model, lbd_set, **kwargs):

    list_sdg_results, bic, ebic = regularization_path(
        estim_fct_with_flag=estim_fct_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        dim_ld=model.DIM_LD,
        N=model.N * (1 + model.J),
        verbatim=True,  # __name__ == "__main__",
        # additional parameter
        model=model,
        **kwargs,
    )

    return RegularizationPathRes(
        multi_run=list_sdg_results,
        argmin_bic=bic[-1].argmin(),
        bic=bic,
        ebic=ebic,
        lbd_set=lbd_set,
    )


# ====================================================== #
