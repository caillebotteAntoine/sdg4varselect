"""
Module for abstract class AbstractCoxModel.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, W0221

from abc import abstractmethod
import functools
import parametrization_cookbook.jax as pc

import jax.numpy as jnp
import jax.random as jrd
from jax import jit

from sdg4varselect.models.abstract.abstract_cox_mem_joint_model import (
    AbstractCoxMemJointModel,
)
from sdg4varselect.models.abstract.abstract_mixed_effect_model import (
    AbstractMixedEffectsModel,
)

from sdg4varselect.models.logistic_mixed_effect_model import (
    LogisticMixedEffectsModel,
)


class WeibullCoxMemJointModel(AbstractCoxMemJointModel):
    """define a joint model of an mixed effects model and a cox model with weibull baseline hazard"""

    def __init__(self, mem: type[AbstractMixedEffectsModel], P, a=80, b=35, **kwargs):
        AbstractCoxMemJointModel.__init__(self, mem=mem, P=P, a=a, b=b, **kwargs)

        self.init()

    def init(self):
        """here you define the parametrization of the model
        and don't forget to call the mother init function at the end"""
        self._parametrization = pc.NamedTuple(
            alpha=pc.Real(scale=10), beta=pc.Real(scale=1, shape=(self.P,))
        )

        AbstractCoxMemJointModel.init(self)

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"WCoxMemJM_N{self.N}_J{self.J}_P{self.P}"

    # ============================================================== #
    @abstractmethod
    @functools.partial(jit, static_argnums=0)
    def likelihood_only_prior(self, params, **kwargs) -> jnp.ndarray:
        """return likelihood with only the gaussian prior"""
        return jnp.array([0])

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
        """Function that return the log of the baseline hazard"""
        out = jnp.log(b / a) + (b - 1) * jnp.log(times / a)
        assert out.shape == times.shape
        return out

    # ============================================================== #


def create_logistic_weibull_jm(N, J, P, **kwargs):
    mem_model = LogisticMixedEffectsModel(N=N, J=J)
    model = WeibullCoxMemJointModel(mem_model, P=P)
    return model


def get_params_star(model):

    return model.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=110.11,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(model.P - 4,))]
        ),
    )


if __name__ == "__main__":
    from sdg4varselect.plot import plot_sample

    myModel = create_logistic_weibull_jm(100, 5, 10)

    my_params_star = myModel.new_params(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        alpha=110.1,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )

    myobs, mysim = myModel.sample(
        my_params_star, jrd.PRNGKey(0), weibull_censoring_loc=77
    )

    _, _ = plot_sample(myobs, mysim, my_params_star, censoring_loc=77, a=80, b=35)
