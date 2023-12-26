# Create by antoine.caillebotte@inrae.fr

import parametrization_cookbook.jax as pc
import numpy as np


import jax.numpy as jnp
import jax.random as jrd
from jax import jit, jacfwd
import functools

from collections import namedtuple


from sdg4varselect.data_handler import Data_handler
from sdg4varselect.simulation import (
    mem_simulation,
    cov_simulation,
    cox_simulation,
    gaussian_prior,
)


# ===================================================== #
# ====================== LOGISTIC ===================== #
# ===================================================== #
@jit
def logistic_curve_float(x, supremum: float, midpoint: float, growth_rate: float):
    return supremum / (1 + jnp.exp(-(x - midpoint) / growth_rate))


@jit
def logistic_curve(
    time: jnp.ndarray,  # shape = (J,) [None, :]
    supremum: jnp.ndarray,  # shape = (N,) [:,None]
    midpoint: jnp.ndarray,  # shape = (N,) [:,None]
    growth_rate: jnp.ndarray,  # shape = (N,) [:,None]
) -> jnp.ndarray:  # shape = (N,J)
    return supremum[:, None] / (
        1 + jnp.exp(-(time - midpoint[:, None]) / growth_rate[:, None])
    )


@jit
def log_hazard(
    time: jnp.ndarray,  # shape = (N,num)
    phi1: jnp.ndarray,  # shape = (N,)
    phi2: jnp.ndarray,  # shape = (N,)
    mu3: jnp.ndarray,  # shape = (1,)
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

    logistic_value = logistic_curve(time, phi1, phi2, jnp.array([mu3]))
    assert logistic_value.shape == time.shape

    log_h_0 = jnp.log(b / a) + (b - 1) * jnp.log(time / a)
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
        "a": 80,  # params_star_weibull.a,  #
        "b": 35,  # params_star_weibull.b,  #
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
    assert time.shape == (N, J)
    assert phi1.shape == (N,)
    assert phi2.shape == (N,)

    pred = logistic_curve(
        time, supremum=phi1, midpoint=phi2, growth_rate=jnp.array([params.mu3])
    )  # shape = (N,J)

    likelihood_nlmem = -J / 2 * jnp.log(2 * jnp.pi * params.sigma2) - jnp.nansum(
        (Y - pred) ** 2, axis=1
    ) / (2 * params.sigma2)

    assert likelihood_nlmem.shape == (N,)
    return likelihood_nlmem


class Logistic_model:
    def __init__(self, DIM_HD):
        self._parametrization = pc.NamedTuple(
            mu1=pc.RealPositive(scale=0.5),
            mu2=pc.Real(scale=100),
            mu3=pc.RealPositive(scale=5),
            gamma2_1=pc.RealPositive(scale=0.001),
            gamma2_2=pc.RealPositive(scale=10),
            sigma2=pc.RealPositive(scale=0.001),
            alpha=pc.Real(scale=10),
            beta=pc.Real(scale=1, shape=(DIM_HD,)),
        )

    @property
    def parametrization(self):
        return self._parametrization

    def reals1d_to_hstack_params(self, theta_reals1d):
        return jnp.hstack(list(self._parametrization.reals1d_to_params(theta_reals1d)))

    def new_params(self, **kwargs):
        theta_reals1d = self._parametrization.params_to_reals1d(**kwargs)
        return self._parametrization.reals1d_to_params(theta_reals1d)

    # ============================================================== #

    @functools.partial(jit, static_argnums=0)
    def likelihood_array(self, theta_reals1d, **kwargs):
        """return likelihood"""
        params = self._parametrization.reals1d_to_params(theta_reals1d)

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

    @functools.partial(jit, static_argnums=0)
    def likelihood(self, theta_reals1d, **kwargs):
        return self.likelihood_array(theta_reals1d, **kwargs).sum()

    @functools.partial(jit, static_argnums=0)
    def jac_likelihood(self, theta_reals1d, **kwargs):
        return jacfwd(self.likelihood_array)(theta_reals1d, **kwargs)

    # ============================================================== #


# ===================================================== #
# ================== DATA GENERATION ================== #
# ===================================================== #
def sample_logistic_model(
    params_star,
    PRNGKey,
    N_IND,
    J_OBS,
    weibull_censoring_loc,
):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""

    def nlmem_simulation(params, PRNGKey, N_IND, J_OBS, t_min, t_max):
        """return longitudinal simulation
        and latente variable simulation in the two dict"""

        def logistic_fct(time, phi1, phi2, phi3):
            return logistic_curve(
                time=time,
                supremum=phi1,
                midpoint=phi2,
                growth_rate=phi3,
            )

        random_effects = {"phi1": ("mu1", "gamma2_1"), "phi2": ("mu2", "gamma2_2")}
        fixed_effets = {"phi3": "mu3"}

        time = jnp.linspace(60, 135, num=J_OBS)
        time = jnp.concatenate(jnp.array([[time]] * N_IND), axis=0)
        PRNGKey, key = jrd.split(PRNGKey, num=2)
        time += jrd.uniform(key, minval=-2, maxval=2, shape=time.shape)

        obs = {"time": time}

        obs2, sim, PRNGKey = mem_simulation(
            params,
            PRNGKey,
            N_IND,
            "sigma2",
            logistic_fct,
            random_effects,
            fixed_effets,
            fct_kwargs={"time": obs["time"]},
        )

        obs.update(obs2)

        return obs, sim, PRNGKey

    def cox_weibull_simulation(params, PRNGKey, N_IND, logistic_sim):
        baseline_kwargs = {"a": 80, "b": 35}

        def baseline_fct(t, a, b):
            return b / a * (t / a) ** (b - 1)

        link_kwargs = {
            "phi1": logistic_sim["phi1"],
            "phi2": logistic_sim["phi2"],
            "mu3": params.mu3,
        }

        def link_fct(t, alpha, phi1, phi2, mu3):
            return alpha * logistic_curve_float(t, phi1, phi2, mu3)

        DIM_COV = params.beta.shape[0]
        cov, PRNGKey = cov_simulation(PRNGKey, min=-1, max=1, shape=(N_IND, DIM_COV))

        _, sim, PRNGKey = cox_simulation(
            params,
            PRNGKey,
            cov @ params.beta,
            baseline_fct,
            baseline_kwargs,
            link_fct,
            link_kwargs,
        )

        return {"cov": cov}, sim, PRNGKey

    # ============================================================== #

    obs, sim, PRNGKey = nlmem_simulation(
        params_star, PRNGKey, N_IND, J_OBS, t_min=60, t_max=135
    )

    obs2, sim2, PRNGKey = cox_weibull_simulation(
        params_star, PRNGKey, N_IND, logistic_sim=sim
    )

    obs.update(obs2)
    sim.update(sim2)

    rng = np.random.default_rng()
    C = weibull_censoring_loc * rng.weibull(35, len(sim["T uncensored"]))
    # C = np.minimum(C, obs["time"].max() + 1)

    T = np.minimum(sim["T uncensored"], C)
    delta = sim["T uncensored"] < C

    obs.update({"T": T, "delta": delta})
    sim["C"] = C

    # obs["Y"] = np.array(
    #     [np.where(obs["time"][i] < T[i], obs["Y"][i], np.nan) for i in range(len(T))]
    # )
    # obs["time"] = np.array(
    #     [np.where(obs["time"][i] < T[i], obs["time"][i], np.nan) for i in range(len(T))]
    # )

    return obs, sim, PRNGKey


def get_params_star(DIM_HD):
    model = Logistic_model(DIM_HD)
    params_star = model.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=11.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(DIM_HD - 4,))]
        ),
    )
    return params_star


def sample_one(PRNGKey, N_IND, J_OBS, DIM_HD, weibull_censoring_loc):
    params_star = get_params_star(DIM_HD)

    obs, _, _ = sample_logistic_model(
        params_star,
        PRNGKey,
        N_IND,
        J_OBS,
        weibull_censoring_loc,
    )

    dh = Data_handler()
    dh.add_data(**obs)

    return dh


# ============================================================== #


if __name__ == "__main__":
    N_IND = 100
    J_OBS = 5
    DIM_COV = 4

    from work import sdgplt

    model = Logistic_model(DIM_COV)

    params_star = model.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=11.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    PRNGKey = jrd.PRNGKey(0)

    def test_censoring_loc(censoring_loc, PRNGKey):
        obs, sim, PRNGKey = sample_logistic_model(
            params_star,
            PRNGKey=PRNGKey,
            N_IND=N_IND,
            J_OBS=J_OBS,
            weibull_censoring_loc=censoring_loc,
        )
        _, _ = sdgplt.plot_sample(obs, sim, params_star, censoring_loc, 80, 35)

        return obs, sim, PRNGKey

    obs, sim, PRNGKey = test_censoring_loc(1000, PRNGKey)  # a = 80, b = 35
    obs, sim, PRNGKey = test_censoring_loc(85, PRNGKey)  # ~20%
    obs, sim, PRNGKey = test_censoring_loc(80.5, PRNGKey)  # ~40%
    obs, sim, PRNGKey = test_censoring_loc(77, PRNGKey)  # ~60%
    obs, sim, PRNGKey = test_censoring_loc(73, PRNGKey)  # ~80%
