# Create by antoine.caillebotte@inrae.fr

from collections import namedtuple
from functools import partial

from sdg4varselect import jacrev, jit, jnp, jacfwd
from sdg4varselect.pk_model import gaussian_prior, pk_curve

# ====================== PARAMIETRIZATION ===================== #
# ============================================================= #
params = namedtuple(
    "params",
    ("mu1", "mu2", "V", "gamma2_1", "gamma2_2", "sigma2"),
)
# ==================================================== #
# ==================== LIKELIHOOD ==================== #
# ==================================================== #


@jit
def log_hazard(
    time: jnp.ndarray,  # shape = (N,num)
    phi1: jnp.ndarray,  # shape = (N,)
    phi2: jnp.ndarray,  # shape = (N,)
    mu3: jnp.ndarray,  # shape = (1,)
    D: jnp.ndarray,  # shape = (1,)
    a: jnp.ndarray,  # shape = (1,)
    b: jnp.ndarray,  # shape = (1,)
    beta: jnp.ndarray,  # shape = (p,)
    alpha: jnp.ndarray,  # shape = (1,)
    cov: jnp.ndarray,  # shape = (N,p)
) -> jnp.ndarray:  # shape = (N, num)
    """hazard(t) = h0(t) * exp(beta^T U  + alpha*m(t))
    with : h0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}

    return : log(b/a) + (b-1)*log(t/a) + beta^T U + alpha*m(t)
    """

    logistic_value = pk_curve(time, D=D, ka=phi1, Cl=phi2, V=jnp.array([mu3]))
    assert logistic_value.shape == time.shape

    log_h_0 = jnp.log(b / a)
    log_h_0 += (b - 1) * jnp.log(time / a)
    assert log_h_0.shape == time.shape

    beta_prod_cov = (cov @ beta)[:, None]
    assert beta_prod_cov.shape[0] == log_h_0.shape[0]

    out = log_h_0 + alpha * logistic_value
    return beta_prod_cov + out


@jit
def likelihood_survival_without_prior(
    params, phi1, phi2, T, delta, cov, **kwargs
) -> jnp.ndarray:
    """return likelihood without the gaussian prior"""
    (N,) = T.shape
    (p,) = params.beta.shape
    assert T.shape == (N,)
    assert delta.shape == (N,)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)
    assert cov.shape == (N, p)
    # ===================== #
    # === survival_likelihood === #
    # ===================== #
    # survival_likelihood = log(survival_fct) + log(hazard_fct)

    # ================= survival_fct ================= #
    # log_survival_fct = - int_0^T hazard(s) ds
    time_s = jnp.linspace(0, T, num=100)[1:].T

    hazard_kwargs = {
        "time": time_s,
        "phi1": phi1,
        "phi2": phi2,
        "mu3": params.mu3,
        "D": -100,
        "a": 35,  # params_star_weibull.a,params.a,  #
        "b": 15,  # params_star_weibull.b,params.b,  #
        "alpha": params.alpha,
        "beta": params.beta,
        "cov": cov,
    }
    log_hazard_value = log_hazard(**hazard_kwargs)
    assert time_s.shape == log_hazard_value.shape

    log_survival_fct = -jnp.trapz(jnp.exp(log_hazard_value), time_s)
    assert log_survival_fct.shape == (N,)
    # =============== end survival_fct =============== #

    # ================= hazard_fct ================= #
    # log_hazard_fct = delta * log(b*a^-b * T^{b-1}) + beta^T U + alpha*m(T, phi_g)
    # Comme time_s[:,-1] == T, on peut faire :
    log_hazard_fct = log_hazard_value[:, -1]
    assert log_hazard_fct.shape == (N,)
    # =============== end hazard_fct =============== #

    return delta * log_hazard_fct + log_survival_fct


# ============================================================== #
@jit
def likelihood_nlmem_without_prior(
    params, Y, time, phi1, phi2, **kwargs
) -> jnp.ndarray:
    """return likelihood without the gaussian prior"""
    N, J = Y.shape
    assert time.shape == (J,)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)

    pred = pk_curve(
        time, D=-100, ka=phi1, Cl=phi2, V=jnp.array([params.mu3])
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


# jacfwd is more efficient for tall matrix
# jacrev is more efficient for wide matrix

un_jit_jac_likelihood = jacfwd(likelihood_array)


@partial(
    jit,
    static_argnums=(1),
)
def jac_likelihood(theta_reals1d, parametrization, **kwargs):
    return un_jit_jac_likelihood(theta_reals1d, parametrization, **kwargs)
