"""
Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp


from results.logistic_model.one_estim import one_estim

from sdg4varselect.models import WeibullCoxJM, logisticMEM
from sdg4varselect.outputs import MultiRunRes

import sdg4varselect.plot as sdgplt
from sdg4varselect.models.linear_model import LinearModel


from sdg4varselect.algo import get_GDFIM_settings
from sdg4varselect.algo.gradient_descent_fim import GradientDescentFIM as GD_FIM


def one_estim_P(P):
    myModel = WeibullCoxJM(
        logisticMEM(N=1000, J=15), P=P, alpha_scale=0.001, a=800, b=10
    )

    p_star = myModel.new_params(
        mean_latent={"mu1": 200, "mu2": 500},
        mu3=150,
        cov_latent=jnp.diag(jnp.array([40, 100])),
        var_residual=100,
        alpha=0.005,
        beta=jnp.concatenate(  # jnp.zeros(shape=(myModel.P,)),  #
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=7700)

    return one_estim(jrd.PRNGKey(0), myModel, myobs, lbd=0.01, save_all=True)


set_p = jnp.array([5, 10, 12, 14, 16, 25, 30, 50, 150, 200, 250, 300])
# res = MultiRunRes([one_estim_P(p) for p in set_p])

myModel = WeibullCoxJM(logisticMEM(N=1000, J=15), P=300, alpha_scale=0.001, a=800, b=10)
res = MultiRunRes.load(myModel, filename_add_on="computation_time_estim_light")


chrono = jnp.array([r.chrono.total_seconds() / 60 for r in res])
lm = LinearModel(chrono.shape[0])
algo_settings = get_GDFIM_settings(preheating=400, heating=600)
algo = GD_FIM(1000, algo_settings)

obs = {"time": set_p / 300, "Y": chrono}

lm_res = []
for i in range(10):
    theta0 = jrd.normal(jrd.PRNGKey(i), shape=(lm.parametrization.size,))
    lm_res.append(algo.fit(lm, obs, theta0))
lm_res = MultiRunRes(lm_res)

sdgplt.plot(lm_res)


sdgplt.FIGSIZE = 10
fig = sdgplt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(set_p, obs["Y"], ".")
ax.plot(set_p, lm_res[0].last_theta[0] + lm_res[0].last_theta[1] / 300 * set_p)

ax.set_xlabel("P")
ax.set_ylabel("Estimation times in minutes")
