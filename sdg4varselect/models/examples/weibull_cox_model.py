"""
Module for abstract class AbstractCoxModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221

import functools


import jax.numpy as jnp
import jax.random as jrd
from jax import jit
import parametrization_cookbook.jax as pc

from sdg4varselect.models.abstract.abstract_cox_model import (
    AbstractCoxModel,
)


class WeibullCoxModel(AbstractCoxModel):
    def __init__(self, N, P, a_scale=10, b_scale=10, **kwargs):
        AbstractCoxModel.__init__(self, N=N, P=P, **kwargs)

        self._parametrization = pc.NamedTuple(
            a=pc.RealPositive(scale=a_scale),
            b=pc.RealPositive(scale=b_scale),
            beta=pc.Real(scale=1, shape=(P,)),
        )

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"WCoxM_N{self.N}_P{self.P}"

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(
        self,
        params,
        times,  # shape = (N,num)
        **kwargs,
    ):
        """Function that return the log of the baseline hazard

        log(b/a* (t/a)^(b-1)) = log(b/a) + (b-1) log(t/a)
        """
        out = jnp.log(params.b / params.a) + (params.b - 1) * jnp.log(times / params.a)
        assert out.shape == times.shape
        return out

    # ============================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    myModel = WeibullCoxModel(N=5000, P=1, a_scale=20)

    p_star = myModel.new_params(
        a=80,
        b=35,
        beta=jnp.array([0]),
        # jnp.concatenate(
        #     [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        # ),
    )

    myobs, mysim = myModel.sample(
        p_star,
        jrd.PRNGKey(0),
        weibull_censoring_loc=7700,
    )

    # ============================================================== #
    from sdg4varselect.outputs import MultiRunRes
    from sdg4varselect.algo import SGD_FIM, get_GDFIM_settings
    from sdg4varselect.learning_rate import create_multi_step_size
    import sdg4varselect.plot as sdgplt

    algo_settings = get_GDFIM_settings(preheating=500, heating=600, learning_rate=1e-12)
    step_sizes = create_multi_step_size(list(algo_settings))
    labels = ["Gradient step size", "Jacobian step size", "Fisher step size"]

    sdgplt.figure()
    _ = [lr.plot(label=labels[i]) for i, lr in enumerate(step_sizes)]
    sdgplt.plt.legend()
    _ = sdgplt.plt.xlim(0, 1200)

    def one_estim(prngkey, model, data, save_all=True):
        prngkey_theta, prngkey_estim = jrd.split(prngkey)
        theta0 = 0.2 * jrd.normal(prngkey_theta, shape=(model.parametrization.size,))

        algo = SGD_FIM(prngkey_estim, 10000, algo_settings)
        res = algo.fit(
            model, data, theta0, ntry=5, partial_fit=False, save_all=save_all
        )

        return res  # , algo

    multi_estim = MultiRunRes(
        [one_estim(jrd.PRNGKey(key), myModel, myobs, save_all=True) for key in range(1)]
    )
    # === PLOT === #
    sdgplt.FIGSIZE = 13

    _ = sdgplt.plot(
        multi_estim,
        params_star=myModel.hstack_params(p_star),
        params_names=myModel.params_names,
    )

    print(multi_estim.chrono)

    # ============================================================== #
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(
        [myobs["T"], mysim["T uncensored"]],  # , sim["C"]],
        bins=20,
        density=True,
        label=["censored survival time", "survival time"],  # , "censuring time"],
    )

    def weibull_fct(t, a, b):
        return b / a * (t / a) ** (b - 1) * np.exp(-((t / a) ** b))

    t = np.linspace(
        myobs["T"].min(), max(myobs["T"].max(), mysim["T uncensored"].max()), num=100
    )
    ax.plot(t, weibull_fct(t, 80, 35), label="weibull baseline")

    ax.plot(
        t,
        weibull_fct(t, 7700, 35),
        label="censured time weibull distribution",
    )
    ax.plot(
        t,
        weibull_fct(t, multi_estim[0].last_theta[0], multi_estim[0].last_theta[1]),
        label="estimated weibull distribution",
    )
    ax.legend()
