"""
Module for LearningRate class.

Create by antoine.caillebotte@inrae.fr
TODO : documentation
"""

# pylint: disable-all

import dataclasses

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit

import matplotlib.pyplot as plt


class _StepSettings:
    def __init__(self, step: float = 0, coef: float = 1):

        self.step = step

        if not isinstance(coef, (int, float)):
            raise TypeError("coef must be int or float")
        self.coef = coef

    def __repr__(self) -> str:
        return f"step : {self.step}, coef = {self.coef}"

    @property
    def step(self) -> float:
        """return step"""
        return self._step

    @step.setter
    def step(self, step) -> None:
        if step is None:
            step = jnp.nan
        elif not isinstance(step, (int, float)):
            raise TypeError("step must be int or float")
        self._step = step


@dataclasses.dataclass
class LearningRateSettings:  # pylint:disable = C0115
    preheating: _StepSettings = _StepSettings(0, 1)
    heating: _StepSettings = _StepSettings(None, 1)

    @classmethod
    def new(
        cls,
        preheating: float = 0,
        coef_preheating: float = 1,
        heating: float = None,
        coef_heating: float = 1,
    ) -> "LearningRateSettings":  # pylint:disable = C0116
        return cls(
            preheating=_StepSettings(preheating, coef_preheating),
            heating=_StepSettings(heating, coef_heating),
        )

    def __repr__(self) -> str:
        return f"preheating({self.preheating}), heating({self.heating})"


@jit
def _preheating_value(x, step, coef):  # pylint:disable = C0103
    return jax.lax.cond(
        step == 0, lambda s: 0.0, lambda s: 1 * jnp.exp(coef * (1 - s / step)), x
    )


@jit
def _heating_value(x, step, coef):
    return jax.lax.cond(
        x == step, lambda s: 0.0, lambda s: 1 / jnp.pow(s - step, coef), x
    )


@jit
def call(
    step: int, preheating_step, preheating_coef, heating_step, heating_coef, max
) -> float:  # pylint:disable = C0116, R0913
    return max * jnp.select(
        [
            step < preheating_step,
            step >= 1 + heating_step,  # ~jnp.isnan(self._heating) and
        ],
        [
            _preheating_value(step, preheating_step, preheating_coef),
            _heating_value(step, heating_step, heating_coef),
        ],
        default=1.0,
    )


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
        preheating: float = 0,
        coef_preheating: float = 1,
        heating: float = None,
        coef_heating: float = 1,
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
        # if not isinstance(settings, LearningRateSettings):
        #     raise TypeError("preheating must be LearningRateSettings")
        self._settings = LearningRateSettings.new(
            preheating, coef_preheating, heating, coef_heating
        )

        self._max = 1

    def __repr__(self) -> str:
        return f"Learning rate [{self._settings}, max = {self._max}]"

    # === PROPERTY === #
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
    def preheating(self) -> _StepSettings:
        """return preheating"""
        return self._settings.preheating

    @property
    def heating(self) -> _StepSettings:
        """return heating"""
        return self._settings.heating

    def __call__(self, step: int) -> float:
        return call(
            step,
            self.preheating.step,
            self.preheating.coef,
            self.heating.step,
            self.heating.coef,
            self._max,
        )

    def plot(self, label=None):
        if jnp.isnan(self.heating.step):
            if self.preheating.step == 0:
                x = np.linspace(0, 200)
            else:
                x = np.linspace(
                    0, 5 * self.preheating.step, num=4 * self.preheating.step
                )
        else:
            x = np.linspace(0, 5 * self.heating.step, num=4 * self.heating.step)

        y = [self(int(i)) for i in x]

        if label is None:
            return plt.plot(x, y)
        return plt.plot(x, y, label=label)


_default_step_size_settings = {
    "coef_heating": 0.65,
    "preheating": 1000,
    "heating": 3500,
    "coef_preheating": float(jnp.log(1e-8)),
}

default_step_size = LearningRate(**_default_step_size_settings)


cst_step_size = LearningRate(
    preheating=0, coef_heating=1, heating=None, coef_preheating=1
)
