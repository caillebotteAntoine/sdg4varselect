"""
Module that define functions to perform multiple selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116, W0221
import sys
import functools

from copy import copy

from jax import jit
import jax.numpy as jnp
import jax.random as jrd

import parametrization_cookbook.jax as pc

from results.simulation_study.multi_res import add_flag

from sdg4varselect.models import (
    AbstractMixedEffectsModel as AbstractMEM,
    AbstractHDModel,
    cov_simulation,
)

from sdg4varselect.learning_rate import LearningRate
from sdg4varselect.algo import SPGD_FIM
from sdg4varselect.exceptions import sdg4vsNanError
import sdg4varselect.algo.preconditioner as preconditioner

# N = int(sys.argv[2])
# P = int(sys.argv[3])
# algo_name = sys.argv[4]
# seed = int(sys.argv[1])

N = 200
P = 10
J = 15
algo_name = "Fisher"
seed = 0


class LogisticMixedEffectsModel(AbstractMEM, AbstractHDModel):
    """define a logistic mixed effects model"""

    def __init__(self, N=1, J=1, P=1, **kwargs):
        AbstractHDModel.__init__(self, P=P)
        AbstractMEM.__init__(self, N=N, J=J, me_name=["phi1", "phi2"], **kwargs)

    @property
    def name(self):
        return f"LogisticMEM_N{self.N}_J{self.J}_P{self.P}"

    def init_parametrization(self):
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

    def sample(self, params_star, prngkey, **kwargs):
        """Sample one data set for the model"""

        (prngkey_time, prngkey_mem, prngkey_cov) = jrd.split(prngkey, num=3)

        # === nlmem_simulation() === #
        time = jnp.linspace(100, 1800, self.J)
        time = jnp.tile(time, (self.N, 1))
        time += 10 * jrd.uniform(prngkey_time, minval=-2, maxval=2, shape=time.shape)

        cov = cov_simulation(prngkey_cov, cov_min=-1, cov_max=1, shape=(self.N, self.P))

        obs, sim = AbstractMEM.sample(
            self, params_star, prngkey_mem, mem_obs_time=time, cov=cov
        )

        return {"mem_obs_time": time, "cov": cov} | obs, sim


# ====================================================== #
# ====================================================== #
# ====================================================== #
def algo_factory(model, prngkey, lbd):
    learning_rate = float(jnp.log(1e-8))

    if lbd is not None:
        step_size = LearningRate(
            coef_heating=0.65,
            preheating=100,
            heating=500,
            coef_preheating=learning_rate,
        )

    else:
        step_size = LearningRate(
            coef_heating=0.65,
            preheating=0,
            heating=100,
            coef_preheating=learning_rate,
        )

    step_size_approx_sto = copy(step_size)
    step_size_approx_sto.heating = None

    step_size_fisher = copy(step_size_approx_sto)
    step_size_fisher.max = 0.9

    precond = preconditioner.Fisher(
        step_size_approx_sto=step_size_approx_sto, step_size_fisher=step_size_fisher
    )
    algo = SPGD_FIM(
        prngkey,
        5000 if lbd is not None else 500,
        step_size,
        precond,
        lbd=lbd,
        alpha=1.0,
        threshold=1e-2,
    )
    return algo


def results(*args, **kwargs):
    res, flag = one_estim(*args, **kwargs)

    return res, res[-1].theta_reals1d[-1, :]


# ====================================================== #

myModel = LogisticMixedEffectsModel(N=N, J=15, P=P)

print(f"P = {myModel.P}, N = {myModel.N}")


p_star = myModel.new_params(
    mean_latent={"mu1": 100, "mu2": 1200},
    cov_latent=jnp.diag(jnp.array([50, 1000])),
    tau=150,
    var_residual=30,
    beta=jnp.concatenate(
        [jnp.array([300, 100, -200]), jnp.zeros(shape=(myModel.P - 3,))]
    ),
)

mylbd_set = 10 ** jnp.linspace(-2, 2, num=5)

myprngkey = jrd.PRNGKey(seed)
print(f"seed = {seed}, prngkey = {myprngkey}")


mydata, sim = myModel.sample(p_star, jrd.PRNGKey(1))


prngkey_list = jrd.split(myprngkey, num=len(mylbd_set))
# out = []
# for i, lbd in enumerate(mylbd_set):
#     if i >= 2:
#         res, theta0 = results(
#             theta0=theta0,
#             prngkey=prngkey_list[i],
#             data=mydata,
#             lbd=lbd,
#             save_all=True,
#         )
#         print(res.chrono, jnp.prod(jnp.array(res.theta.shape[:-1])))
#         out.append(res)


# ====================================================== #
def one_estim(prngkey, data, lbd=None, save_all=True):
    # data = mydata
    # prngkey = myprngkey
    # save_all = True

    data = copy(data)

    prngkey_lasso, prngkey_estim = jrd.split(prngkey)

    # ===== LASSO ===== #
    model = LogisticMixedEffectsModel(N=N, J=J, P=P)
    theta0 = 0.2 * jrd.normal(myprngkey, shape=(model.parametrization.size,))

    algo = algo_factory(model, myprngkey, lbd=mylbd_set[2])
    # =================== MCMC configuration ==================== #
    algo.init_mcmc(theta0, model, sd={"phi1": 5, "phi2": 50})

    for var_lat in algo.latent_variables.values():
        var_lat.adaptative_sd = True
    # ==================== END configuration ==================== #

    lasso = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)

    # ======================== SHRINK COV ======================= #
    selected_component = (lasso.theta_reals1d[-1] != 0).at[: -model.P].set(True)
    hd_selected = selected_component[-P:]
    new_p = int(hd_selected.sum())

    data["cov"] = data["cov"][:, hd_selected]
    theta0 = lasso.theta_reals1d[-1, selected_component]
    # ===== ESTIMATION ===== #
    model = LogisticMixedEffectsModel(N=N, J=J, P=new_p)
    # algo = algo_factory(model, lbd=None)
    algo.update_mcmc(model)
    algo.lbd = None
    algo.max_iter = 200

    estim = algo.fit(model, data, theta0, ntry=5, partial_fit=False, save_all=save_all)
    estim.expand_theta((lasso.last_theta != 0).at[:-P].set(True))

    sdgplt.plot_mcmc(algo.latent_variables)

    return MultiRunRes([lasso, estim])


def estim_with_flag(*args, **kwargs):
    """must return the estimation results and
    a flag which indicates if the regularization path is finished"""
    res_estim = one_estim(*args**kwargs)
    flag = (res_estim[-1].last_theta[-P:] != 0).sum() == 0
    return res_estim, flag


star_p = myModel.hstack_params(p_star)
p_names = [
    "$\\mu_1$",
    "$\\mu_2$",
    "$\\tau$",
    "$\\gamma^2_1$",
    "$\\gamma^2_{12}$",
    "$\\gamma^2_{21}$",
    "$\\gamma^2_2$",
    "$\\sigma^2$",
] + [f"$\\beta_{1+i}$" for i in range(myModel.P)]


from sdg4varselect.outputs import MultiRunRes
import sdg4varselect.new_plot as sdgplt

res_estim = one_estim(myprngkey, mydata, lbd=mylbd_set[3], save_all=True)

_ = sdgplt._plot_theta(
    MultiRunRes(res_estim).theta.T,
    star_p,
    p_names,
    id_to_plot=[0, 1, 2, 3, 6, 7] + [8 + i for i in range(myModel.P)],
    fig=sdgplt.figure(),
)


# print(lasso.chrono, estim.chrono)

# from sdg4varselect import regularization_path, lasso_into_estim
# from sdg4varselect.outputs import RegularizationPathRes, MultiRunRes

# list_sdg_results, bic, ebic = regularization_path(
#     estim_fct_with_flag=estim_with_flag,
#     prngkey=prngkey,
#     lbd_set=lbd_set,
#     P=P,
#     N=N * (1 + J),
#     verbatim=True,  # __name__ == "__main__",
#     # additional parameter
#     model=model,
#     **kwargs,
# )
