"""
Module for abstract class AbstractJointModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221
import sys

from abc import abstractmethod
import functools


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
import parametrization_cookbook.jax as pc

from sdg4varselect.models import (
    AbstractMixedEffectsModel,
    AbstractHDModel,
    cov_simulation,
)

import sdg4varselect.algo.preconditioner as sdgpreconditioner

from results.simulation_study.multi_res import add_flag, one_result
from sdg4varselect.algo import SPGD_FIM, get_gdfim_settings
from sdg4varselect.exceptions import sdg4vsNanError
import sdg4varselect.plot as sdgplt

N = int(sys.argv[2])
P = int(sys.argv[3])
algo_name = sys.argv[4]
seed = int(sys.argv[1])

# N = 200
# P = 30
# algo_name = "Fisher"
# seed = 0


class HDLogisticMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):
    """define a logistic mixed effects model"""

    def __init__(self, N=1, J=1, P=1, **kwargs):
        super().__init__(
            N=N,
            J=J,
            me_name=["phi1", "phi2"],
            **kwargs,
        )
        AbstractHDModel.__init__(self, P=P)
        self.init()

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"LogisticMEM_N{self.N}_J{self.J}_P{self.P}"

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                mu1=pc.RealPositive(scale=100),
                mu2=pc.RealPositive(scale=2000),
            ),
            tau=pc.RealPositive(scale=100),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(100, 2000)),
            var_residual=pc.RealPositive(scale=100),
            beta=pc.Real(scale=10, shape=(self.P,)),
        )
        AbstractHDModel.init_dim(self)

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def mixed_effect_function(
        self,
        params,
        times: jnp.ndarray,  # shape = (J,) [None, :]
        phi1: jnp.ndarray,  # shape = (N,) [:,None]
        phi2: jnp.ndarray,  # shape = (N,) [:,None]
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        phi1 = supremum
        phi2 = midpoint
        tau = growth rate
        """
        ksi = cov @ params.beta + phi2

        out = phi1[:, None] / (1 + jnp.exp(-(times - ksi[:, None]) / params.tau))
        assert out.shape == times.shape
        return out

    # ============================================================== #

    @abstractmethod
    def sample(
        self,
        params_star,
        prngkey,
        **kwargs,
    ):
        """Sample one data set for the model"""

        (prngkey_time, prngkey_mem, prngkey_cov) = jrd.split(prngkey, num=3)

        # === nlmem_simulation() === #
        time = jnp.linspace(100, 1800, self.J)
        time = jnp.tile(time, (self.N, 1))
        time += 10 * jrd.uniform(prngkey_time, minval=-2, maxval=2, shape=time.shape)

        cov = cov_simulation(prngkey_cov, cov_min=-1, cov_max=1, shape=(self.N, self.P))

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time, cov=cov
        )

        return {"mem_obs_time": time, "cov": cov} | obs, sim


# ====================================================== #
# ====================================================== #
# ====================================================== #


def algo_factory(name: str, p: int):  # , preheating, heating, learning_rate):
    algo_settings = get_gdfim_settings(
        preheating=3000, heating=3500, learning_rate=1e-8
    )

    if name.lower() == "fisher":
        preconditioner = sdgpreconditioner.Fisher(list(algo_settings)[1:])

    elif name == "adagrad":
        algo_settings = [
            {
                "learning_rate": 1,
                "preheating": 0,
                "heating": 3500,
                "max": 1e-2,
            }
        ]

        preconditioner = sdgpreconditioner.AdaGrad(regularization=1e-5)
    elif name == "fisheradagrad":

        preconditioner = sdgpreconditioner.FisherAdaGrad(
            P=p, settings=list(algo_settings)[1:], regularization=1e-5
        )
    else:
        raise ValueError("algo name must be fisheradagrad, fisher or adagrad")

    return algo_settings, preconditioner


@add_flag
def one_estim_with_flag(prngkey, model, data, algo_name, lbd=None, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    algo_settings, preconditioner = algo_factory(algo_name, model.P)
    algo = SPGD_FIM(
        prngkey_estim, 5000, algo_settings, preconditioner, lbd=lbd, alpha=1.0
    )
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"phi1": 5, "phi2": 50})

    for var_lat in algo.latent_variables.values():
        var_lat.adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    return res


# ====================================================== #

myModel = HDLogisticMixedEffectsModel(N=N, J=15, P=P)

print(f"P = {myModel.P}, N = {myModel.N}")


p_star = myModel.new_params(
    mean_latent={"mu1": 100, "mu2": 1200},
    cov_latent=jnp.diag(jnp.array([50, 2000])),
    tau=150,
    var_residual=30,
    beta=jnp.concatenate([jnp.array([90, 80, 70]), jnp.zeros(shape=(myModel.P - 3,))]),
)

mylbd_set = 10 ** jnp.linspace(-2, 0, num=40)

myprngkey = jrd.PRNGKey(seed)
print(f"seed = {seed}, prngkey = {myprngkey}")


mydata, _ = myModel.sample(p_star, myprngkey)


if __name__ == "__main__":
    try:
        estim_res = one_result(
            one_estim_with_flag,
            myprngkey,
            myModel,
            data=mydata,
            algo_name=algo_name,
            lbd_set=mylbd_set,
            save_all=False,
        )

        estim_res.save(
            myModel, root="files_unmerged", filename_add_on=f"lbd40_S{seed}_{algo_name}"
        )

    except sdg4vsNanError as err:
        print(f"{err} :  estimation cancelled !")
