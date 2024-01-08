# Create by antoine.caillebotte@inrae.fr

import parametrization_cookbook.jax as pc
import numpy as np


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
import functools

from sdg4varselect.data_handler import Data_handler
from sdg4varselect.Joint_model import (
    JointModel,
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


# ============================================================== #


def mixed_effect_function(
    times,  # shape = (N,num)
    phi1,  # shape = (N,)
    phi2,  # shape = (N,)
    mu3,
    **kwargs,
):
    out = logistic_curve(times, phi1, phi2, jnp.array([mu3]))
    assert out.shape == times.shape
    return out


def log_baseline_hazard(
    times,  # shape = (N,num)
    a: jnp.ndarray,  # shape = (1,)
    b: jnp.ndarray,  # shape = (1,)
    **kwargs,
):
    out = jnp.log(b / a) + (b - 1) * jnp.log(times / a)
    assert out.shape == times.shape
    return out


class Logistic_JM(JointModel):
    def __init__(self, N=1, J=1, DIM_HD=1):
        super().__init__(
            N, J, DIM_HD, log_baseline_hazard, mixed_effect_function, a=80, b=35
        )

        self._parametrization = pc.NamedTuple(
            mu1=pc.RealPositive(scale=0.5),
            mu2=pc.RealPositive(scale=100),
            mu3=pc.RealPositive(scale=5),
            gamma2_1=pc.RealPositive(scale=0.001),
            gamma2_2=pc.RealPositive(scale=10),
            sigma2=pc.RealPositive(scale=0.001),
            alpha=pc.Real(scale=10),
            beta=pc.Real(scale=1, shape=(DIM_HD,)),
        )

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
            + self.likelihood_mem_without_prior(params, mu3=params.mu3, **kwargs)
            + self.likelihood_survival_without_prior(params, **kwargs)
        )


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
            return logistic_curve(time, phi1, phi2, phi3)

        random_effects = {"phi1": ("mu1", "gamma2_1"), "phi2": ("mu2", "gamma2_2")}
        fixed_effets = {"phi3": "mu3"}

        time = jnp.linspace(60, 135, num=J_OBS)
        time = jnp.concatenate(jnp.array([[time]] * N_IND), axis=0)
        PRNGKey, key = jrd.split(PRNGKey, num=2)
        time += jrd.uniform(key, minval=-2, maxval=2, shape=time.shape)

        obs = {"mem_obs_time": time}

        obs2, sim, PRNGKey = mem_simulation(
            params,
            PRNGKey,
            N_IND,
            "sigma2",
            logistic_fct,
            random_effects,
            fixed_effets,
            fct_kwargs={"time": obs["mem_obs_time"]},
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
    model = Logistic_JM(DIM_HD=DIM_HD)
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


def sample_one(PRNGKey, model, weibull_censoring_loc):
    params_star = get_params_star(model.DIM_HD)

    obs, _, _ = sample_logistic_model(
        params_star,
        PRNGKey,
        model.N,
        model.J,
        weibull_censoring_loc,
    )

    dh = Data_handler()
    dh.add_data(**obs)

    return dh


# ============================================================== #


if __name__ == "__main__":
    from work import sdgplt

    model = Logistic_JM(N=100, J=5, DIM_HD=4)

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
            N_IND=model.N,
            J_OBS=model.J,
            weibull_censoring_loc=censoring_loc,
        )
        _, _ = sdgplt.plot_sample(obs, sim, params_star, censoring_loc, 80, 35)

        return obs, sim, PRNGKey

    obs, sim, PRNGKey = test_censoring_loc(1000, PRNGKey)  # a = 80, b = 35
    obs, sim, PRNGKey = test_censoring_loc(85, PRNGKey)  # ~20%
    obs, sim, PRNGKey = test_censoring_loc(80.5, PRNGKey)  # ~40%
    obs, sim, PRNGKey = test_censoring_loc(77, PRNGKey)  # ~60%
    obs, sim, PRNGKey = test_censoring_loc(73, PRNGKey)  # ~80%
