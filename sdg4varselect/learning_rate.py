"""
Module for LearningRate class.

Create by antoine.caillebotte@inrae.fr
"""

import functools

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit


class LearningRate:
    """
    Class for managing learning rates with dynamic behavior.

    Methods:
    ----------
    - zero():
        Static method to create a learning rate instance with preheating=1000, max=0.
    - one():
        Static method to create a learning rate instance with preheating=0, max=1.
    - from_0_to_1(heat, coef_preheating):
        Static method to create a learning rate instance for transitioning from 0 to 1.
    - from_1_to_0(burnin, coef_heating):
        Static method to create a learning rate instance for transitioning from 1 to 0.
    - __call__(step: int) -> float: Calculate the learning rate for a given iteration.
    - __repr__() -> str: Return a string representation of the learning rate configuration.
    - plot(label=None): Plot the learning rate curve.

    Properties:
    ----------
    - preheating: Get the preheating value.
    - coef_preheating: Get the coefficient for preheating exponential growth.
    - heating: Get the heating value.
    - coef_heating: Get the coefficient for heating power-law decay.
    - max: Get the maximum learning rate.
    - step_flat: Get the number of iterations for a flat learning rate.
    """

    def __init__(
        self,
        preheating: int = 0,
        coef_preheating: float = 1,
        heating: int = None,
        coef_heating: float = 1,
        value_max: float = 1,
        step_flat: int = 0,
        *args,
        **kwargs
    ):
        """Initializes a new Learning rate object with the given parameters.

        Parameters:
        ----------
            preheating (int): Number of iterations for preheating.
            coef_preheating (float): Coefficient for preheating exponential growth.
            heating (int or None): Number of iterations for heating or None for no heating.
            coef_heating (float): Coefficient for heating power-law decay.
            value_max (float): Maximum learning rate.
            step_flat (int): Number of iterations for a flat learning rate.
        """
        if not isinstance(preheating, int):
            raise TypeError("preheating must be int")
        self._preheating = preheating

        if not isinstance(coef_preheating, (int, float)):
            raise TypeError("coef_preheating must be int or float")
        self._coef_preheating = coef_preheating

        self.heating = heating

        if not isinstance(coef_heating, (int, float)):
            raise TypeError("coef_heating must be int or float")
        self._coef_heating = coef_heating

        self.max = value_max

        if not isinstance(step_flat, (int, float)):
            raise TypeError("step_flat must be int or float")
        self._step_flat = step_flat

    # === PROPERTY === #
    @property
    def preheating(self):
        """return preheating"""
        return self._preheating

    @property
    def coef_preheating(self):
        """return coef_preheating"""
        return self._coef_preheating

    @property
    def heating(self):
        """return heating"""
        return self._heating

    @heating.setter
    def heating(self, heating) -> None:
        if heating is None:
            self._heating = jnp.nan
        else:
            if not isinstance(heating, int):
                raise TypeError("heating must be int")
            self._heating = heating - 1

    @property
    def coef_heating(self):
        """return preheating"""
        return self._coef_heating

    @property
    def max(self):
        """return max"""
        return self._max

    @max.setter
    def max(self, value_max) -> None:
        if not isinstance(value_max, (int, float)):
            raise TypeError("value_max must be int or float")
        self._max = value_max

    @property
    def step_flat(self):
        """return step_flat"""
        return self._step_flat

    # === STATIC METHOD === #
    @staticmethod
    def zero():
        return LearningRate(preheating=1000, value_max=0)

    @staticmethod
    def one():
        return LearningRate(preheating=0, value_max=1)

    @staticmethod
    def from_0_to_1(heat, coef_preheating):
        return LearningRate(heat, coef_preheating, 100000, 1)

    @staticmethod
    def from_1_to_0(burnin, coef_heating):
        return LearningRate(0, 1, burnin, coef_heating)

    @functools.partial(jit, static_argnums=0)
    def _preheating_value(self, step):
        return jax.lax.cond(
            self._preheating == 0,
            lambda s: 0.0,
            lambda s: self._max
            * jnp.exp(self._coef_preheating * (1 - s / self._preheating)),
            step,
        )
        # return jax.lax.cond(
        #     step < self._preheating,
        #     self._max
        #     * np.exp(self._coef_preheating * (1 - float(step) / self._preheating)),
        #     0,
        # )

    @functools.partial(jit, static_argnums=0)
    def _heating_value(self, step):
        return jax.lax.cond(
            step == self._heating,
            lambda s: 0.0,
            lambda s: self._max / jnp.pow(s - self._heating, self._coef_heating),
            step,
        )
        # return jax.lax.cond(
        #     self._heating is not None and step >= 1 + self._heating,
        #     ,
        #     0,
        # )

    def __call__(self, step: int) -> float:
        return jnp.select(
            [
                step < self._step_flat,
                step < self._preheating,
                step >= 1 + self._heating,  # ~jnp.isnan(self._heating) and
            ],
            [
                0,
                self._preheating_value(step),
                self._heating_value(step),
            ],
            default=self._max,
        )

        # if step < self._step_flat:  # ensures zero value before growth
        #     return 0

        # if step < self._preheating:  # before exp(coeff *(1-step/preheating))
        #     return self._max * np.exp(
        #         self._coef_preheating * (1 - float(step) / self._preheating)
        #     )

        # if self._heating is not None:
        #     if step >= 1 + self._heating:  # after (step - heating)^-coef
        #         return self._max * pow(step - self._heating, -self._coef_heating)

        # return self._max

    def __repr__(self) -> str:
        max_msg = str(self._max) + "*"
        if self._max == 1:
            max_msg = ""
        out = (
            self.__class__.__name__
            + " :"
            + "\n\t i ->\t | "
            + max_msg
            + "exp("
            + str(self._coef_preheating)
            + "*(1-i/"
            + str(self._preheating)
            + "))\t if i < "
            + str(self._preheating)
            + "\n\t\t | "
            + max_msg
            + "( i - "
            + str(self._coef_heating)
            + ")^"
            + str(-self._coef_heating)
            + "\t if i >= "
            + str(self._heating)
            + "\n\t\t | "
            + str(self._max)
            + "\t otherwise"
        )

        return out

    def plot(self, label=None):
        import matplotlib.pyplot as plt

        if jnp.isnan(self.heating):
            if self.preheating == 0:
                x = np.linspace(0, 200)
            else:
                x = np.linspace(0, 5 * self.preheating, num=4 * self.preheating)
        else:
            x = np.linspace(0, 5 * self.heating, num=4 * self.heating)

        y = [self.__call__(i) for i in x]

        if label is None:
            return plt.plot(x, y)
        return plt.plot(x, y, label=label)


def create_multi_step_size(settings: list[dict], num_step_size: int = 3):
    """create num_step_size LearningRate with settings as parameter"""
    # !! check if setting is list of settings set !!
    settings = list(settings)
    # if not isinstance(settings, list):
    #     raise TypeError("settings must be a list of dict !")

    if len(settings) < 0:
        raise TypeError(
            "settings must contain at least one parameter set to define a step sequence !"
        )

    default_keys = ["learning_rate", "preheating", "heating", "max"]

    for setting in settings:
        if not all([x in setting for x in default_keys]):
            raise TypeError(
                "settings set must contain this four keys : learning_rate, preheating, heating, max"
            )

    while len(settings) < num_step_size:
        settings.append(settings[0])

    while len(settings) > num_step_size:
        settings.pop()

    step_size = []
    for setting in settings:
        setting["coef_burnin"] = 0.65
        setting["coef_preheating"] = float(np.log(setting["learning_rate"]))
        setting["value_max"] = float(setting["max"])
        step_size.append(LearningRate(**setting))

    return step_size


if __name__ == "__main__":
    f = LearningRate(10, -2, 20, 0.75)
    print(f)
    x = np.arange(40, dtype=float)
    y = [f(i) for i in x]
    print(y)

    import matplotlib.pyplot as plt

    plt.step(x, y)
    plt.show()

    f = LearningRate.from_0_to_1(10, -2)
    y = [f(i) for i in range(20)]

    f = LearningRate.from_0_to_1(10, -2)
    y += [f(i) for i in range(20, 40)]

    plt.step(x, y)
    plt.show()

    print(type(f))
    print(isinstance(f, LearningRate))

    f = LearningRate(10, -2, 20, 0.75, step_flat=5)
    f.plot()

    plt.figure()
    f = LearningRate(10, -2)
    f.plot()

    plt.figure()
    f = LearningRate.one()
    f.plot()

    plt.figure()
    f = create_multi_step_size(
        [
            {
                "learning_rate": 1,
                "preheating": 0,
                "heating": None,
                "max": 1e-1,
            }
        ]
    )
    f[0].plot()
