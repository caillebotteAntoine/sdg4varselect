import jax.numpy as jnp
import jax.random as jrd
import scipy.special

from sdg4varselect.miscellaneous import step_message
from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.outputs import RegularizationPathRes


def eBIC(theta_HD, log_likelihood, n):
    """
    eBIC = k*ln(n) - 2*ln(L) + 2*ln(C^p_k)

    where :
        - k is the number of parameter estimated (ie non zero parameter in HD parameter)
        - n is the sample size
        - L the maximzed value of the likelihood function
    """

    k = (theta_HD != 0).sum(axis=1)
    assert k.shape == log_likelihood.shape
    ebic_pen = scipy.special.binom(theta_HD.shape[1], k)
    assert ebic_pen.shape == log_likelihood.shape

    return -2 * log_likelihood + k * jnp.log(n) + 2 * jnp.log(ebic_pen)


def BIC(theta_HD, log_likelihood, n):
    """
    BIC = k*ln(n) - 2*ln(L)

    where :
        - k is the number of parameter estimated (ie non zero parameter in HD parameter)
        - n is the sample size
        - L the maximzed value of the likelihood function
    """
    k = (theta_HD != 0).sum(axis=1)
    assert k.shape == log_likelihood.shape

    return -2 * log_likelihood + k * jnp.log(n)


def _regularization_path(
    estim_fct_with_flag, prngkey, lbd_set, verbatim=False, *args, **kwargs
):
    prngkey_list = jrd.split(prngkey, num=len(lbd_set))

    def iter_estim():
        for i, lbd in enumerate(lbd_set):
            if verbatim:
                print(step_message(i, len(lbd_set)), end="\r")

            kwargs["lbd"] = lbd
            kwargs["prngkey"] = prngkey_list[i]
            res_estim, flag_selection = estim_fct_with_flag(*args, **kwargs)

            if flag_selection:
                for _ in range(len(lbd_set) - i):
                    yield res_estim
                break
            else:
                yield res_estim

    return [res for res in iter_estim()]


def regularization_path(
    estim_fct_with_flag, prngkey, lbd_set, dim_ld, N, verbatim=False, *args, **kwargs
):
    # === VARIABLE SELECTION === #
    list_sdg_results = _regularization_path(
        estim_fct_with_flag=estim_fct_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        verbatim=verbatim,
        # additional parameter
        **kwargs
    )
    if list_sdg_results is None:
        raise sdg4vsNanError

    bic = jnp.array(
        [
            BIC(
                jnp.array([r.last_theta[dim_ld:] for r in res]),
                jnp.array([r.likelihood for r in res]),
                N,
            )
            for res in list_sdg_results
        ]
    ).T

    return RegularizationPathRes(
        listSDGResults=list_sdg_results,
        bic=bic,
    )
