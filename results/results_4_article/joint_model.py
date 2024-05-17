"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116, W0221

import sys
import functools

from jax import jit
import jax.numpy as jnp
import jax.random as jrd

import parametrization_cookbook.jax as pc

from multi_res import add_flag, one_result

from sdg4varselect.models import AbstractMixedEffectsModel, WeibullCoxJM
from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings
from sdg4varselect.exceptions import sdg4vsNanError


class LogisticMixedEffectsModel(AbstractMixedEffectsModel):
    """define a logistic mixed effects model"""

    def __init__(self, N=1, J=1, **kwargs):
        AbstractMixedEffectsModel.__init__(
            self,
            N=N,
            J=J,
            me_name=["phi1", "phi2"],
            **kwargs,
        )

        self.init()

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"LogisticMEM_N{self.N}_J{self.J}"

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""

        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                asymptotic=pc.RealPositive(scale=100),
                inflexion=pc.Real(loc=100, scale=100),
            ),
            tau=pc.RealPositive(scale=100),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(200, 200)),
            var_residual=pc.RealPositive(scale=100),
        )

        AbstractMixedEffectsModel.init(self)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi1: jnp.ndarray,  # shape = (N,) [:,None]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = asymptotic
        phi2 = inflexion
        """

        out = phi1[:, None] / (1 + jnp.exp(-(times - phi2[:, None]) / params.tau))
        assert out.shape == times.shape
        return out

    # ============================================================== #

    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ):
        """Sample one data set for the model"""

        (
            prngkey_time,
            prngkey_mem,
        ) = jrd.split(prngkey, num=2)

        # === nlmem_simulation() === #
        time = jnp.linspace(100, 1500, self.J)
        time = jnp.tile(time, (self.N, 1))
        time += 10 * jrd.uniform(prngkey_time, minval=-2, maxval=2, shape=time.shape)

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time
        )

        return {"mem_obs_time": time} | obs, sim


algo_settings = get_GDFIM_settings(preheating=600, heating=1000, learning_rate=1e-6)


@add_flag
def one_estim_with_flag(prngkey, model, data, lbd=None, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    algo = SPGD_FIM(prngkey_estim, 10000, algo_settings, lbd=lbd, alpha=1.0)
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"phi1": 5, "phi2": 20})

    for var_lat in algo.latent_variables.values():
        var_lat.adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    return res


# ====================================================== #


# joint model with coxModel is all ready implement in sdg4varselect for all MixedEffectsModel
myModel = WeibullCoxJM(
    mem=LogisticMixedEffectsModel(N=int(sys.argv[2]), J=15), P=int(sys.argv[3]), alpha_scale=0.001, a=800, b=10
)

print(f"P = {myModel.P}, N = {myModel.N}")


p_star = myModel.new_params(
    mean_latent={"asymptotic": 200, "inflexion": 500},
    tau=150,
    cov_latent=jnp.diag(jnp.array([40, 100])),
    var_residual=10,
    alpha=0.05,
    beta=jnp.concatenate(
        [jnp.array([-3, -2, 2, 3]), jnp.zeros(shape=(myModel.P - 4,))]
    ),
)


mylbd_set = 10 ** jnp.linspace(-1.5, -0.1, num=10)  # P = 500

seed = int(sys.argv[1])
myprngkey = jrd.PRNGKey(seed)
print(f"seed = {seed}, prngkey = {myprngkey}")


mydata, _ = myModel.sample(p_star, myprngkey, weibull_censoring_loc=7700)

try:
    estim_res = one_result(
        one_estim_with_flag,
        myprngkey,
        myModel,
        data=mydata,
        lbd_set=mylbd_set,
        save_all=False,
    )

    estim_res.save(myModel, root="files_unmerged", filename_add_on=f"S{seed}")

except sdg4vsNanError as err:
    print(f"{err} :  estimation cancelled !")


# ====================================================== #
