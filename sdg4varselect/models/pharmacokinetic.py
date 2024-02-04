# pylint: disable=C0116, W0221

import functools
import numpy as np
import parametrization_cookbook.jax as pc


import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect._data_handler import DataHandler
from sdg4varselect.models.abstract_joint_model import (
    AbstractJointModel,
    mem_simulation,
    cov_simulation,
    cox_simulation,
    gaussian_prior,
)


# ===================================================== #
# ====================== LOGISTIC ===================== #
# ===================================================== #


@jit
def pk_curve_float(t, D: float, ka: float, Cl: float, V: float):
    return D * ka / (V * ka - Cl) * (jnp.exp(-ka * t) - jnp.exp(-Cl / V * t))


@jit
def pk_curve(
    time: jnp.ndarray,  # shape = (N,J)
    D: jnp.ndarray,  # shape = (1,) [:,None]
    ka: jnp.ndarray,  # shape = (N,) [:,None]
    Cl: jnp.ndarray,  # shape = (N,) [:,None]
    V: jnp.ndarray,  # shape = (N,) [:,None]
) -> jnp.ndarray:  # shape = (N,J)
    return (
        D
        * ka[:, None]
        / (V[:, None] * (ka[:, None] - Cl[:, None] / V[:, None]))
        * (jnp.exp(-ka[:, None] * time) - jnp.exp(-Cl[:, None] / V[:, None] * time))
    )


# ============================================================== #


class PharmaJM(AbstractJointModel):
    def __init__(self, N=1, J=1, DIM_HD=1):
        super().__init__(N, J, DIM_HD, a=35, b=15, D=-100)

        self._parametrization = pc.NamedTuple(
            mu1=pc.RealPositive(scale=10),
            mu2=pc.RealPositive(scale=5),
            mu3=pc.RealPositive(scale=50),
            gamma2_1=pc.RealPositive(scale=1),
            gamma2_2=pc.RealPositive(scale=1),
            sigma2=pc.RealPositive(scale=0.001),
            alpha=pc.Real(scale=1),
            beta=pc.Real(scale=1, shape=(DIM_HD,)),
        )

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(
        self,
        times,  # shape = (N,num)
        a: jnp.ndarray,  # shape = (1,)
        b: jnp.ndarray,  # shape = (1,)
        **kwargs,
    ):
        out = jnp.log(b / a) + (b - 1) * jnp.log(times / a)
        assert out.shape == times.shape
        return out

    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi1: jnp.ndarray,  # shape = (N,) [:,None]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        mu3: jnp.ndarray,  # shape = ()
        D,
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        mu3 = growth_rate
        """
        out = (
            D
            * phi1[:, None]
            / (mu3 * (phi1[:, None] - phi2[:, None] / mu3))
            * (jnp.exp(-phi1[:, None] * times) - jnp.exp(-phi2[:, None] / mu3 * times))
        )

        assert out.shape == times.shape
        return out

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
        prngkey,
        weibull_censoring_loc,
    ):
        """return longitudinal and survival simulation
        and latente variable simulation in the two dict"""

        # def pk_fct(time, phi1, phi2, phi3, D):
        #     return pk_curve(time, D, ka=phi1, Cl=phi2, V=phi3)

        (
            prngkey_mem,
            prngkey_cov,
            prngkey_cox,
            prngkey_weibull,
        ) = jrd.split(prngkey, num=4)

        # === nlmem_simulation() === #
        time = jnp.repeat(
            jnp.exp(jnp.linspace(-3, 4, num=self.J))[jnp.newaxis, :],
            self.N,
            axis=0,
        )

        obs = {"mem_obs_time": time}

        obs_mem, sim = mem_simulation(
            params_star,
            prngkey_mem,
            N_IND=self.N,
            noise_variance="sigma2",
            fct=lambda time, phi1, phi2, phi3, D: self.mixed_effect_function(
                time, phi1, phi2, phi3, D
            ),
            random_effects={
                "phi1": ("mu1", "gamma2_1"),
                "phi2": ("mu2", "gamma2_2"),
            },
            fixed_effets={"phi3": "mu3"},
            fct_kwargs={"time": obs["mem_obs_time"], "D": -100},
        )

        obs.update(obs_mem)

        # === cox_weibull_simulation === #
        link_kwargs = {
            "phi1": sim["phi1"],
            "phi2": sim["phi2"],
            "mu3": params_star.mu3,
            "D": -100,
        }

        cov = cov_simulation(prngkey_cov, min=-1, max=1, shape=(self.N, self.DIM_HD))

        _, sim_cox = cox_simulation(
            params_star,
            prngkey_cox,
            cov @ params_star.beta,
            baseline_fct=lambda t, a, b: b / a * (t / a) ** (b - 1),
            baseline_kwargs={"a": 35, "b": 15},
            link_fct=lambda t, alpha, phi1, phi2, mu3, D: alpha
            * pk_curve_float(t, D, ka=phi1, Cl=phi2, V=mu3),
            link_kwargs=link_kwargs,
        )

        obs.update({"cov": cov})
        sim.update(sim_cox)

        # ============================================================== #
        C = jrd.weibull_min(
            prngkey_weibull, weibull_censoring_loc, 35, shape=sim["T uncensored"].shape
        )

        T = np.minimum(sim["T uncensored"], C)
        delta = sim["T uncensored"] < C

        obs.update({"T": T, "delta": delta})
        sim["C"] = C

        return obs, sim


def get_params_star(dim_hd):
    model = PharmaJM(DIM_HD=dim_hd)
    params_star = model.new_params(
        mu1=8,
        mu2=6,
        mu3=40,
        gamma2_1=0.2,
        gamma2_2=0.1,
        sigma2=1e-3,
        alpha=100.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(dim_hd - 4,))]
        ),
    )
    return params_star


# ============================================================== #


if __name__ == "__main__":
    from sdg4varselect.plot import plot_sample

    myModel = PharmaJM(N=100, J=10, DIM_HD=5)

    my_params_star = myModel.new_params(
        mu1=8,
        mu2=6,
        mu3=40,
        gamma2_1=0.2,
        gamma2_2=0.1,
        sigma2=1e-3,
        alpha=11.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.DIM_HD - 4,))]
        ),
    )

    def test_censoring_loc(censoring_loc):
        myobs, mysim = myModel.sample(
            my_params_star, jrd.PRNGKey(0), weibull_censoring_loc=censoring_loc
        )
        _, _ = plot_sample(myobs, mysim, my_params_star, censoring_loc, 35, 15)

    # test_censoring_loc(1000)  # a = 80, b = 35
    # test_censoring_loc(85)  # ~20%
    # test_censoring_loc(80.5)  # ~40%
    # test_censoring_loc(77)  # ~60%
    # test_censoring_loc(73)  # ~80%

    theta0 = 0.2 * jrd.normal(jrd.PRNGKey(0), shape=(myModel.parametrization.size, 100))
    params0 = jnp.array(
        [myModel.reals1d_to_hstack_params(theta0[:, i]) for i in range(100)]
    )
    print(params0.mean(axis=0))
