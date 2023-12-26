import numpy as np


class learning_rate:
    def __init__(
        self,
        preheating: int = 0,
        coef_preheating: float = 1,
        heating: int = None,
        coef_heating: float = 1,
        max: float = 1,
        step_flat: int = 0,
        *args,
        **kwargs
    ):
        if not isinstance(preheating, int):
            raise TypeError("preheating must be int")
        self._preheating = preheating

        if not isinstance(coef_preheating, (int, float)):
            raise TypeError("coef_preheating must be int or float")
        self._coef_preheating = coef_preheating

        if heating is None:
            self._heating = None
        else:
            if not isinstance(heating, int):
                raise TypeError("heating must be int")
            self._heating = heating - 1

        if not isinstance(coef_heating, (int, float)):
            raise TypeError("coef_heating must be int or float")
        self._coef_heating = coef_heating

        if not isinstance(max, (int, float)):
            raise TypeError("max must be int or float")
        self._max = max

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

    @property
    def coef_heating(self):
        """return preheating"""
        return self._coef_heating

    @property
    def max(self):
        """return max"""
        return self._max

    @property
    def step_flat(self):
        """return step_flat"""
        return self._step_flat

    # === STATIC METHOD === #
    @staticmethod
    def zero():
        return learning_rate(preheating=1000, max=0)

    @staticmethod
    def one():
        return learning_rate(preheating=0, max=1)

    @staticmethod
    def from_0_to_1(heat, coef_preheating):
        return learning_rate(heat, coef_preheating, 10**10, 1)

    @staticmethod
    def from_1_to_0(burnin, coef_heating):
        return learning_rate(0, 1, burnin, coef_heating)

    def __call__(self, iter: int) -> float:
        if iter < self._step_flat:  # ensures zero value before growth
            return 0

        if iter < self._preheating:  # before exp(coeff *(1-iter/preheating))
            return self._max * np.exp(
                self._coef_preheating * (1 - float(iter) / self._preheating)
            )

        if self._heating is not None:
            if iter >= 1 + self._heating:  # after (iter - heating)^-coef
                return self._max * pow(iter - self._heating, -self._coef_heating)

        return self._max

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

        if self.heating is None:
            if self.preheating == 0:
                x = np.linspace(0, 200)
            else:
                x = np.linspace(0, 2 * self.preheating, num=4 * self.preheating)
        else:
            x = np.linspace(0, 2 * self.heating, num=4 * self.heating)

        y = [self.__call__(i) for i in x]

        if label is None:
            return plt.plot(x, y)
        return plt.plot(x, y, label=label)


def create_multi_step_size(settings, num_step_size=3):
    # !! check if setting is list of settings set !!
    if not isinstance(settings, list):
        raise TypeError("settings must be a list of dict !")

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

    step_size = []
    for setting in settings:
        setting["coef_burnin"] = 0.65
        setting["coef_preheating"] = float(np.log(setting["learning_rate"]))
        step_size.append(learning_rate(**setting))

    return step_size


if __name__ == "__main__":
    f = learning_rate(10, -2, 20, 0.75)
    print(f)
    x = np.arange(40, dtype=float)
    y = [f(i) for i in x]
    print(y)

    import matplotlib.pyplot as plt

    plt.step(x, y)
    plt.show()

    f = learning_rate.from_0_to_1(10, -2)
    y = [f(i) for i in range(20)]

    f = learning_rate.from_0_to_1(10, -2)
    y += [f(i) for i in range(20, 40)]

    plt.step(x, y)
    plt.show()

    print(type(f))
    print(isinstance(f, learning_rate))

    f = learning_rate(10, -2, 20, 0.75, step_flat=5)
    f.plot()

    plt.figure()
    f = learning_rate(10, -2)
    f.plot()

    plt.figure()
    f = learning_rate.one()
    f.plot()
