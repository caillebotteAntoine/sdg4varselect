"""
Module for LearningRate class.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable-all

import functools

import dataclasses

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit

import matplotlib.pyplot as plt


@dataclasses.dataclass
class _StepSettings:
    def __init__(self, step: float = 0, coef: float = 1):

        if step is None:
            step = jnp.nan
        elif not isinstance(step, (int, float)):
            raise TypeError("step must be int or float")
        self.step = step

        if not isinstance(coef, (int, float)):
            raise TypeError("coef must be int or float")
        self.coef = coef


@dataclasses.dataclass
class LearningRateSettings:
    preheating: _StepSettings = _StepSettings(0, 1)
    heating: _StepSettings = _StepSettings(None, 1)

    def __init__(
        self,
        preheating: float = 0,
        coef_preheating: float = 1,
        heating: float = None,
        coef_heating: float = 1,
    ):
        self.preheating = _StepSettings(preheating, coef_preheating)
        self.heating = _StepSettings(heating, coef_heating)


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
        # settings: LearningRateSettings = LearningRateSettings(0, 1, None, 1),
        # value_max: float = 1,
        # step_flat: int = 0,
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
        self._settings = LearningRateSettings(
            preheating, coef_preheating, heating, coef_heating
        )

        self._max = 1

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

    @functools.partial(jit, static_argnums=0)
    def _preheating_value(self, step):
        return jax.lax.cond(
            self.preheating.step == 0,
            lambda s: 0.0,
            lambda s: self._max
            * jnp.exp(self.preheating.coef * (1 - s / self.preheating.step)),
            step,
        )

    @functools.partial(jit, static_argnums=0)
    def _heating_value(self, step):
        return jax.lax.cond(
            step == self.heating.step,
            lambda s: 0.0,
            lambda s: self._max / jnp.pow(s - self.heating.step, self.heating.coef),
            step,
        )

    def __call__(self, step: int) -> float:
        return jnp.select(
            [
                step < self.preheating.step,
                step >= 1 + self.heating.step,  # ~jnp.isnan(self._heating) and
            ],
            [
                self._preheating_value(step),
                self._heating_value(step),
            ],
            default=self._max,
        )

    def plot(self, label=None):
        if jnp.isnan(self.heating.step):
            if self.preheating == 0:
                x = np.linspace(0, 200)
            else:
                x = np.linspace(
                    0, 5 * self.preheating.step, num=4 * self.preheating.step
                )
        else:
            x = np.linspace(0, 5 * self.heating.step, num=4 * self.heating.step)

        y = [self(i) for i in x]

        if label is None:
            return plt.plot(x, y)
        return plt.plot(x, y, label=label)
