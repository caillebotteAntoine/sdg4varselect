"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116, W0221

import sys
import functools
import pandas as pd

from jax import jit
import jax.numpy as jnp
import jax.random as jrd

import parametrization_cookbook.jax as pc

from results.simulation_study.multi_res import add_flag, one_result

from sdg4varselect.outputs import MultiRunRes

from sdg4varselect.models import (
    AbstractMixedEffectsModel,
    AbstractHDModel,
    cov_simulation,
)
from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings
from sdg4varselect.exceptions import sdg4vsNanError


class LogisticMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):
    """define a logistic mixed effects model"""

    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractMixedEffectsModel.__init__(
            self,
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
                mu1=pc.RealPositive(scale=500),
                mu2=pc.RealPositive(scale=1000),
            ),
            cov_latent=pc.MatrixDiagPosDef(dim=2, scale=(100, 100)),
            var_residual=pc.RealPositive(scale=100),
            alpha=pc.Real(scale=50, shape=(5,)),
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
        cov_keep: jnp.ndarray,  # shape = (N,5)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve"""
        ksi = cov_keep @ params.alpha + cov @ params.beta + phi1

        out = 100 / (1 + jnp.exp(-(times - ksi[:, None]) / phi2[:, None]))
        assert out.shape == times.shape
        return out

    # ============================================================== #


algo_settings = get_GDFIM_settings(preheating=500, heating=1500, learning_rate=1e-6)


@add_flag
def one_estim_with_flag(prngkey, model, data, lbd=None, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    algo = SPGD_FIM(prngkey_estim, 2000, algo_settings, lbd=lbd, alpha=1.0)
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"phi1": 5, "phi2": 20})

    for var_lat in algo.latent_variables.values():
        var_lat.adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    return res


# ====================================================== #


# mydata, _ = myModel.sample(p_star, myprngkey)


data_obs = pd.read_csv("data_obs.csv", sep=";", index_col=0)
data_obs = data_obs.sort_values(by=["GENOTYPE", "Day"])

Y = jnp.array(data_obs.Y).reshape((220, 18))
N, J = Y.shape

time = jnp.array([0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])

cov_obs = (
    pd.read_csv("data_real_chr6A.csv", sep=";", decimal=",", index_col=0)
    .drop(columns="Intercept")
    .sort_index()
)

cov_obs = jnp.array(cov_obs.to_numpy(dtype=jnp.float64))
cov_keep = cov_obs[:, :5]
cov_obs = cov_obs[:, 5:10]
N, P = cov_obs.shape

data = {
    "Y": Y,
    "mem_obs_time": jnp.tile(time, (N, 1)),
    "cov_keep": cov_keep,
    "cov": cov_obs,
}

myModel = LogisticMixedEffectsModel(N=N, J=J, P=P)

print(f"P = {myModel.P}, N = {myModel.N}")


if __name__ == "__main__":
    # seed = 0
    ii = 11  # int(sys.argv[1])
    mylbd_set = 10 ** jnp.linspace(-5, 0, num=20)

    estim_res = []
    for seed in range(3):
        myprngkey = jrd.PRNGKey(seed)
        print(f"seed = {seed}, prngkey = {myprngkey}")

        try:
            estim_res.append(
                one_estim_with_flag(
                    prngkey=myprngkey,
                    model=myModel,
                    data=data,
                    lbd=mylbd_set[ii],
                    save_all=False,
                )[0]
            )
        except sdg4vsNanError as err:
            print(f"{err} :  estimation cancelled !")

    MultiRunRes(estim_res).save(
        myModel, root="files_unmerged", filename_add_on=f"lbd[{ii}]"
    )

# ====================================================== #


estim_res = MultiRunRes.load(
    myModel, root="files_unmerged", filename_add_on=f"lbd[{ii}]"
)

likelihood = estim_res.likelihood[:, -1]
pen_estimate = estim_res.last_theta[:, -1, :]
pen = jnp.abs(pen_estimate[:, myModel.DIM_LD :]).sum(axis=-1)
argmax_pen_estimate = (likelihood - mylbd_set[ii] * pen).argmax(axis=0)


argmax_pen_estimate


# import sdg4varselect.plot as sdgplt

# _ = sdgplt.plot_theta(
#     estim_res.final_result,
#     dim_ld=myModel.DIM_LD - 5,
#     params_names=myModel.params_names,
#     id_to_plot=[0, 1, 2, 5, 6],
# )

# _ = sdgplt.plot_theta_hd(
#     estim_res.final_result, dim_ld=myModel.DIM_LD - 5, params_names=myModel.params_names
# )

# _ = sdgplt.plot_reg_path(reg_res=estim_res, dim_ld=myModel.DIM_LD)
# _ = sdgplt.plot_reg_path(reg_res=estim_res.standardize(), dim_ld=myModel.DIM_LD)

# print(estim_res.final_result.last_theta[-1, :])
