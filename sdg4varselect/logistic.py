# Create by antoine.caillebotte@inrae.fr

import parametrization_cookbook.jax as pc
import numpy as np


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
import functools

from sdg4varselect.data_handler import Data_handler
from sdg4varselect.joint_model import (
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
    times: jnp.ndarray,  # shape = (J,) [None, :]
    phi1: jnp.ndarray,  # shape = (N,) [:,None]
    phi2: jnp.ndarray,  # shape = (N,) [:,None]
    mu3: jnp.ndarray,  # shape = ()
    **kwargs,
) -> jnp.ndarray:  # shape = (N,J)
    """
    phi1 = supremum
    phi2 = midpoint
    mu3 = growth_rate
    """
    out = phi1[:, None] / (1 + jnp.exp(-(times - phi2[:, None]) / mu3))
    assert out.shape == times.shape
    return out


# ============================================================== #
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
        super().__init__(N, J, DIM_HD, log_baseline_hazard, logistic_curve, a=80, b=35)

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
    def sample(
        self,
        params_star,
        PRNGKey,
        weibull_censoring_loc,
    ):
        """return longitudinal and survival simulation
        and latente variable simulation in the two dict"""

        (
            PRNGKey_time,
            PRNGKey_mem,
            PRNGKey_cov,
            PRNGKey_cox,
            PRNGKey_weibull,
        ) = jrd.split(PRNGKey, num=5)

        # === nlmem_simulation() === #
        time = jnp.repeat(jnp.linspace(60, 135, num=self.J)[None, :], self.N, axis=0)
        time += jrd.uniform(PRNGKey_time, minval=-2, maxval=2, shape=time.shape)

        obs = {"mem_obs_time": time}

        obs_mem, sim = mem_simulation(
            params_star,
            PRNGKey_mem,
            N_IND=self.N,
            noise_variance="sigma2",
            fct=lambda time, phi1, phi2, phi3: logistic_curve(time, phi1, phi2, phi3),
            random_effects={
                "phi1": ("mu1", "gamma2_1"),
                "phi2": ("mu2", "gamma2_2"),
            },
            fixed_effets={"phi3": "mu3"},
            fct_kwargs={"time": obs["mem_obs_time"]},
        )

        obs.update(obs_mem)

        # === cox_weibull_simulation === #
        link_kwargs = {
            "phi1": sim["phi1"],
            "phi2": sim["phi2"],
            "mu3": params_star.mu3,
        }

        cov = cov_simulation(PRNGKey_cov, min=-1, max=1, shape=(self.N, self.DIM_HD))

        _, sim_cox = cox_simulation(
            params_star,
            PRNGKey_cox,
            cov @ params_star.beta,
            baseline_fct=lambda t, a, b: b / a * (t / a) ** (b - 1),
            baseline_kwargs={"a": 80, "b": 35},
            link_fct=lambda t, alpha, phi1, phi2, mu3: alpha
            * logistic_curve_float(t, phi1, phi2, mu3),
            link_kwargs=link_kwargs,
        )

        obs.update({"cov": cov})
        sim.update(sim_cox)

        # ============================================================== #
        C = jrd.weibull_min(
            PRNGKey_weibull, weibull_censoring_loc, 35, shape=sim["T uncensored"].shape
        )

        T = np.minimum(sim["T uncensored"], C)
        delta = sim["T uncensored"] < C

        obs.update({"T": T, "delta": delta})
        sim["C"] = C

        return obs, sim


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

    obs, _ = model.sample(params_star, PRNGKey, weibull_censoring_loc)

    dh = Data_handler()
    dh.add_data(**obs)

    return dh


# ============================================================== #


if __name__ == "__main__":
    from sdg4varselect.plot import plot_sample

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
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(model.DIM_HD - 4,))]
        ),
    )

    PRNGKey = jrd.PRNGKey(0)

    def test_censoring_loc(censoring_loc, PRNGKey):
        obs, sim = model.sample(
            params_star, PRNGKey, weibull_censoring_loc=censoring_loc
        )
        _, _ = plot_sample(obs, sim, params_star, censoring_loc, 80, 35)

        return obs, sim, PRNGKey

    obs, sim, PRNGKey = test_censoring_loc(1000, PRNGKey)  # a = 80, b = 35
    obs, sim, PRNGKey = test_censoring_loc(85, PRNGKey)  # ~20%
    obs, sim, PRNGKey = test_censoring_loc(80.5, PRNGKey)  # ~40%
    obs, sim, PRNGKey = test_censoring_loc(77, PRNGKey)  # ~60%
    obs, sim, PRNGKey = test_censoring_loc(73, PRNGKey)  # ~80%
