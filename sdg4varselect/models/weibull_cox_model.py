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
    def __init__(self, N, P, a=80, b=35, **kwargs):
        AbstractCoxModel.__init__(self, N=N, P=P, a=a, b=b, **kwargs)

        self._parametrization = pc.NamedTuple(beta=pc.Real(scale=1, shape=(P,)))

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
        a: jnp.ndarray,  # shape = (1,)
        b: jnp.ndarray,  # shape = (1,)
        **kwargs,
    ):
        """Function that return the log of the baseline hazard"""
        out = jnp.log(b / a) + (b - 1) * jnp.log(times / a)
        assert out.shape == times.shape
        return out

    # ============================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    myModel = WeibullCoxModel(N=100, P=5)

    my_params_star = myModel.new_params(
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.DIMCovCox - 4,))]
        ),
    )

    obs, sim = myModel.sample(my_params_star, jrd.PRNGKey(0), weibull_censoring_loc=77)
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(
        [obs["T"], sim["T uncensored"]],  # , sim["C"]],
        bins=20,
        density=True,
        label=["censored survival time", "survival time"],  # , "censuring time"],
    )

    def weibull_fct(t, a, b):
        return b / a * (t / a) ** (b - 1) * np.exp(-((t / a) ** b))

    t = np.linspace(
        obs["T"].min(), max(obs["T"].max(), sim["T uncensored"].max()), num=100
    )
    ax.plot(t, weibull_fct(t, 80, 35), label="weibull baseline")

    ax.plot(
        t,
        weibull_fct(t, 77, 35),
        label="censured time weibull distribution",
    )
    ax.legend()
