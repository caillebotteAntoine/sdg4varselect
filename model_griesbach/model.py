# Create by antoine.caillebotte@inrae.fr

from collections import namedtuple
from functools import partial

from sdg4varselect import jacrev, jit, jnp, jacfwd
from sdg4varselect.linear_model import gaussian_prior, linear_curve

# ==================================================== #
# ==================== LIKELIHOOD ==================== #
# ==================================================== #


@jit
def log_hazard(
    time: jnp.ndarray,  # shape = (N,num)
    phi1: jnp.ndarray,  # shape = (N,)
    phi2: jnp.ndarray,  # shape = (N,)
    beta_surv: jnp.ndarray,  # shape = (p,)
    beta_long: jnp.ndarray,  # shape = (p,)
    alpha: jnp.ndarray,  # shape = (1,)
    cov_surv: jnp.ndarray,  # shape = (N,p)
    cov_long: jnp.ndarray,  # shape = (N,p)
) -> jnp.ndarray:  # shape = (N, num)
    """hazard(t) = h0(t) * exp(beta^T U  + alpha*m(t))
    with : h0(t) = 1

    return : beta^T U + alpha*m(t)
    """
    beta_prod_cov_long = cov_long @ beta_long
    assert beta_prod_cov_long.shape == phi1.shape

    logistic_value = linear_curve(
        time, intercept=phi1 + beta_prod_cov_long, slope=phi2  #
    )
    assert logistic_value.shape == time.shape

    beta_prod_cov_surv = (cov_surv @ beta_surv)[:, None]
    assert beta_prod_cov_surv.shape[0] == time.shape[0]

    out = alpha * logistic_value
    return beta_prod_cov_surv + out


@jit
def likelihood_survival_without_prior(
    params, phi1, phi2, T, delta, cov_surv, cov_long, **kwargs
) -> jnp.ndarray:
    """return likelihood without the gaussian prior"""
    (N,) = T.shape
    (p_surv,) = params.beta_surv.shape
    (p_long,) = params.beta_long.shape
    assert T.shape == (N,)
    assert delta.shape == (N,)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)
    assert cov_surv.shape == (N, p_surv)
    assert cov_long.shape == (N, p_long)
    # ===================== #
    # === survival_likelihood === #
    # ===================== #
    # survival_likelihood = log(survival_fct) + log(hazard_fct)

    # ================= survival_fct ================= #
    # log_survival_fct = - int_0^T hazard(s) ds
    time_s = jnp.linspace(0, T, num=100)[1:].T

    log_hazard_value = log_hazard(
        time=time_s,
        phi1=phi1,
        phi2=phi2,
        alpha=params.alpha,
        beta_surv=params.beta_surv,
        beta_long=params.beta_long,
        cov_surv=cov_surv,
        cov_long=cov_long,
    )
    assert time_s.shape == log_hazard_value.shape

    log_survival_fct = -jnp.trapz(jnp.exp(log_hazard_value), time_s)
    assert log_survival_fct.shape == (N,)
    # =============== end survival_fct =============== #

    # ================= hazard_fct ================= #
    # log_hazard_fct = delta*(log(b*a^-b * T^{b-1}) + beta^T U + alpha*m(T, phi_g))
    # Comme time_s[:,-1] == T, on peut faire :
    log_hazard_fct = log_hazard_value[:, -1]
    assert log_hazard_fct.shape == (N,)
    # =============== end hazard_fct =============== #

    return delta * log_hazard_fct + log_survival_fct


# ============================================================== #
@jit
def likelihood_nlmem_without_prior(
    params, Y, time, phi1, phi2, cov_long, **kwargs
) -> jnp.ndarray:
    """return likelihood without the gaussian prior"""
    N, J = Y.shape
    assert time.shape == (J,)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)

    beta_prod_cov_long = cov_long @ params.beta_long
    assert beta_prod_cov_long.shape == phi1.shape

    pred = linear_curve(
        time,
        intercept=phi1 + beta_prod_cov_long,
        slope=phi2,
    )  # shape = (N,J)

    likelihood_nlmem = -J / 2 * jnp.log(2 * jnp.pi * params.sigma2) - jnp.nansum(
        (Y - pred) ** 2, axis=1
    ) / (2 * params.sigma2)

    assert likelihood_nlmem.shape == (N,)
    return likelihood_nlmem


# ============================================================== #


@partial(
    jit,
    static_argnums=(1),
)
def likelihood_array(theta_reals1d, parametrization, **kwargs):
    """return likelihood"""
    params = parametrization.reals1d_to_params(theta_reals1d)

    latent_prior = gaussian_prior(
        kwargs["phi1"],
        params.mu1,
        params.gamma2_1,
    ) + gaussian_prior(
        kwargs["phi2"],
        params.mu2,
        params.gamma2_2,
    )

    return (
        latent_prior
        + likelihood_nlmem_without_prior(params, **kwargs)
        + likelihood_survival_without_prior(params, **kwargs)
    )


@partial(
    jit,
    static_argnums=(1),
)
def likelihood(theta_reals1d, parametrization, **kwargs):
    return likelihood_array(theta_reals1d, parametrization, **kwargs).sum()


# jacfwd if more efficient for tall matrix
# jacrev if more efficient for wide matrix

un_jit_jac_likelihood = jacfwd(likelihood_array)


@partial(
    jit,
    static_argnums=(1),
)
def jac_likelihood(theta_reals1d, parametrization, **kwargs):
    return un_jit_jac_likelihood(theta_reals1d, parametrization, **kwargs)
