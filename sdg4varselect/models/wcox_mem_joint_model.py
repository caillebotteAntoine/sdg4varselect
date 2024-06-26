"""
Module for abstract class AbstractCoxModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221

import functools
import parametrization_cookbook.jax as pc

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models import (
    AbstractCoxMemJointModel,
    AbstractMixedEffectsModel,
)


class WeibullCoxMemJointModel(AbstractCoxMemJointModel):
    """define a joint model of an mixed effects model and a cox model with weibull baseline hazard"""

    def __init__(
        self, mem: type[AbstractMixedEffectsModel], P, alpha_scale, a=80, b=35, **kwargs
    ):
        AbstractCoxMemJointModel.__init__(self, mem=mem, P=P, a=a, b=b, **kwargs)
        self._alpha_scale = alpha_scale

        self.init()

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""
        self._parametrization = pc.NamedTuple(
            alpha=pc.Real(scale=self._alpha_scale),
            beta=pc.Real(scale=1, shape=(self.P,)),
        )

        AbstractCoxMemJointModel.init(self)

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"WCoxMemJM_N{self.N}_J{self.J}_P{self.P}"

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def log_baseline_hazard(
        self,
        params,
        times,  # shape = (N,num)
        a: jnp.ndarray,  # shape = (1,)
        b: jnp.ndarray,  # shape = (1,)
        *args,
        **kwargs,
    ):
        """Function that return the log of the baseline hazard

        log(b/a* (t/a)^(b-1)) = log(b/a) + (b-1) log(t/a)

        """
        out = jnp.log(b / a) + (b - 1) * jnp.log(times / a)
        assert out.shape == times.shape
        return out

    # ============================================================== #


if __name__ == "__main__":
    from sdg4varselect.plot import plot_sample
    from sdg4varselect.models import logisticMEM

    myModel = WeibullCoxMemJointModel(
        logisticMEM(N=1000, J=15), P=5, alpha_scale=0.001, a=800, b=10
    )

    p_star = myModel.new_params(
        mean_latent={"mu1": 200, "mu2": 500},
        mu3=150,
        cov_latent=jnp.diag(jnp.array([40, 100])),
        var_residual=100,
        alpha=0.005,
        beta=jnp.zeros(shape=(myModel.P,)),  # jnp.concatenate(  #
        #     [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        # ),
    )

    myobs, mysim = myModel.sample(p_star, jrd.PRNGKey(0), weibull_censoring_loc=7700)

    _, _ = plot_sample(myobs, mysim, p_star, censoring_loc=78, **myModel._cst)

    import sdg4varselect.plot as sdgplt

    sdgplt.FIGSIZE = 8

    d = myobs["delta"]
    T = myobs["T"][d]
    T.sort()

    sdgplt.figure()
    sdgplt.plt.ylim(0, 1.1)
    sdgplt.plot(
        T, jnp.array(T[:, None] >= T).mean(axis=0), color="red", label="1-fct rep estim"
    )

    def cdf(t, a, b):
        return jnp.exp(-((t / a) ** b))

    t = jnp.linspace(T.min(), T.max() * 1.1, num=200)
    sdgplt.plt.plot(t, cdf(t, 80, 35), color="green", label="S = 1- fct rep théo")

    sdgplt.plt.legend()
