"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116, W0211
import functools
from datetime import datetime
import sys

from jax import jit
import jax.numpy as jnp
import jax.random as jrd

import parametrization_cookbook.jax as pc

import sdg4varselect.plot as sdgplt

from sdg4varselect.models import (
    AbstractMixedEffectsModel,
    AbstractHDModel,
    cov_simulation,
)

from sdg4varselect.algo import SPGD_FIM, get_GDFIM_settings

from sdg4varselect import regularization_path, lasso_into_estim
from sdg4varselect.outputs import RegularizationPathRes, MultiRunRes
from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.miscellaneous import step_message


algo_settings = get_GDFIM_settings(preheating=600, heating=1000, learning_rate=1e-6)


class HDLogisticMixedEffectsModel(AbstractMixedEffectsModel, AbstractHDModel):
    """HDLogisticMixedEffectsModel"""

    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractMixedEffectsModel.__init__(
            self,
            N=N,
            J=J,
            me_name=["ksi"],
            **kwargs,
        )
        AbstractHDModel.__init__(self, P=P)

        self.init()

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"HDLogisticMEM_N{self.N}_J{self.J}_P{self.P}"

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""
        self._parametrization = pc.NamedTuple(
            mean_latent=pc.NamedTuple(
                mu=pc.RealPositive(scale=2000),
            ),
            psi1=pc.RealPositive(scale=100),
            psi2=pc.RealPositive(scale=100),
            cov_latent=pc.MatrixSymPosDef(dim=1, scale=(100)),
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
        ksi: jnp.ndarray,  # shape = (N,) [:,None]
        cov: jnp.ndarray,  # shape = (N,p)
        **kwargs,
    ) -> jnp.ndarray:  # shape = (N,J)
        """logistic_curve
        psi1 = supremum
        ksi = midpoint
        psi2 = growth rate
        """
        phi = cov @ params.beta + ksi

        out = params.psi1 / (1 + jnp.exp(-(times - phi[:, None]) / params.psi2))
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
            prngkey_mem,
            prngkey_cov,
        ) = jrd.split(prngkey, num=2)

        # === nlmem_simulation() === #
        time = 150 + jnp.arange(0, self.J - 1) * (3000 - 150) / (self.J - 1)
        time = jnp.repeat(time[None, :], self.N, axis=0)

        cov = cov_simulation(prngkey_cov, cov_min=-1, cov_max=1, shape=(self.N, self.P))

        obs, sim = AbstractMixedEffectsModel.sample(
            self, params_star, prngkey_mem, mem_obs_time=time, cov=cov
        )

        return {"mem_obs_time": time, "cov": cov} | obs, sim


def one_estim(prngkey, model, data, lbd=None, save_all=True):
    prngkey_theta, prngkey_estim = jrd.split(prngkey)
    theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

    algo = SPGD_FIM(prngkey_estim, 10000, algo_settings, lbd=lbd, alpha=1.0)
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"ksi": 20})

    algo.latent_variables["ksi"].adaptative_sd = True
    # ==================== END configuration ==================== #
    res = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    return res


def estim_with_flag(model, **kwargs) -> tuple[MultiRunRes, bool]:
    """must return the estimation results and
    a flag which indicates if the regularization path is finished"""
    res_estim = lasso_into_estim(one_estim, model=model, **kwargs)
    dim_ld = model.DIM_LD
    flag = (res_estim[-1].last_theta[dim_ld:] != 0).sum() == 0

    return res_estim, flag


def one_result(prngkey, model, data, lbd_set, save_all=True):

    list_sdg_results, bic = regularization_path(
        estim_fct_with_flag=estim_with_flag,
        prngkey=prngkey,
        lbd_set=lbd_set,
        dim_ld=model.DIM_LD,
        N=model.N * (1 + model.J),
        verbatim=True,  # __name__ == "__main__",
        # additional parameter
        model=model,
        data=data,
        save_all=save_all,
    )

    argmin_bic = bic[-1].argmin()

    return RegularizationPathRes(
        multi_run=list_sdg_results,
        argmin_bic=argmin_bic,
        bic=bic,
        lbd_set=lbd_set,
    )


# ====================================================== #
def multi_run(prngkey, lbd_set, params_star, model, nrun, save_all=True):
    chrono_start = datetime.now()
    print(f'start at {chrono_start.strftime("%d/%m/%Y %H:%M:%S")}')

    prngkey_list = jrd.split(prngkey, num=nrun)

    estim_res = []
    for k in range(nrun):
        print("run", step_message(k, nrun), end="\n")
        data, _ = model.sample(params_star, prngkey_list[k])

        try:
            estim_res.append(
                one_result(
                    prngkey_list[k],
                    model,
                    data=data,
                    lbd_set=lbd_set,
                    save_all=save_all,
                ),
            )

        except sdg4vsNanError as err:
            print(f"{err} :  estimation cancelled !")

    return MultiRunRes(estim_res)


# ====================================================== #
# ====================================================== #
# ====================================================== #

myHDModel = HDLogisticMixedEffectsModel(N=200, J=10, P=50)

p_star = myHDModel.new_params(
    mean_latent={"mu": 1200},
    psi1=200,
    psi2=300,
    cov_latent=jnp.diag(jnp.array([200])),
    var_residual=30,
    beta=jnp.concatenate(
        [jnp.array([100, 50, 20]), jnp.zeros(shape=(myHDModel.P - 3,))]
    ),
)


mylbd_set = 10 ** jnp.linspace(-1.5, -1, num=10)

seed = int(sys.argv[1])
print(seed)

res = multi_run(
    jrd.PRNGKey(seed),
    mylbd_set,
    p_star,
    myHDModel,
    nrun=5,
    save_all=False,
)
res.save(myHDModel, root="files_unmerged", filename_add_on=f"S{seed}")

if __name__ == "__main__":
    sdgplt.FIGSIZE = 10

    fig = sdgplt.boxplot_estimation(
        res.last_theta[:, 1, : myHDModel.DIM_LD + 4].T,
        hline=myHDModel.hstack_params(p_star)[: myHDModel.DIM_LD + 4],
        labels=myHDModel.params_names[: myHDModel.DIM_LD + 4],
        nrows=2,
        ncols=5,
        fig=sdgplt.figure(height=4, width=15),
    )
    fig.tight_layout()
    _ = fig.suptitle("MLE of the parameter", fontsize=15, y=1.05)
