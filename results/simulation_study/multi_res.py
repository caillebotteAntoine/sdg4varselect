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
        P = model.P
        flag = (res_estim[-1].last_theta[-P:] != 0).sum() == 0

        return res_estim, flag

    return estim_with_flag


def one_result(estim_fct_with_flag, prngkey, model, lbd_set, **kwargs):

    list_sdg_results, bic, ebic = regularization_path(
        estim_fct_with_flag=estim_fct_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        P=model.P,
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
from typing import Callable
import jax.random as jrd

from sdg4varselect.outputs import sdg4vsResults
from sdg4varselect.miscellaneous import step_message


def _regularization_path_new(
    estim_fct_with_flag: Callable[[], tuple[type[sdg4vsResults], bool]],
    prngkey,
    lbd_set,
    verbatim=False,
    **kwargs
) -> MultiRunRes:
    """perform an regularization path according to a given estimation function"""
    prngkey_list = jrd.split(prngkey, num=len(lbd_set))

    def iter_estim():
        for i, lbd in enumerate(lbd_set):
            if verbatim:
                print(
                    step_message(i, len(lbd_set)),
                    end="\r" if __name__ == "__main__" else "\n",
                )

            kwargs["lbd"] = lbd
            kwargs["prngkey"] = prngkey_list[i]
            res_estim, flag_selection = estim_fct_with_flag(**kwargs)

            if flag_selection:
                for _ in range(len(lbd_set) - i):
                    yield res_estim
                break
            else:
                yield res_estim

    return MultiRunRes([res for res in iter_estim()])
