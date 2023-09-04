import numpy as np


class learning_rate:
    def __init__(
        self,
        step_heat: int = 0,
        coef_heat: float = 1,
        step_burnin: int = None,
        coef_burnin: float = 1,
        scale: float = 1,
        step_flat: int = 0,
    ):
        if not isinstance(step_heat, int):
            raise TypeError("step_heat must be int")
        self._step_heat = step_heat

        if not isinstance(coef_heat, (int, float)):
            raise TypeError("coef_heat must be int or float")
        self._coef_heat = coef_heat

        if step_burnin is None:
            self._step_burnin = None
        else:
            if not isinstance(step_burnin, int):
                raise TypeError("step_burnin must be int")
            self._step_burnin = step_burnin - 1

        if not isinstance(coef_burnin, (int, float)):
            raise TypeError("coef_burnin must be int or float")
        self._coef_burnin = coef_burnin

        if not isinstance(scale, (int, float)):
            raise TypeError("scale must be int or float")
        self._scale = scale

        if not isinstance(step_flat, (int, float)):
            raise TypeError("step_flat must be int or float")
        self._step_flat = step_flat

    @property
    def step_heat(self):
        """return step_heat"""
        return self._step_heat

    @property
    def coef_heat(self):
        """return coef_heat"""
        return self._coef_heat

    @property
    def step_burnin(self):
        """return step_burnin"""
        return self._step_burnin

    @property
    def coef_burnin(self):
        """return step_heat"""
        return self._coef_burnin

    @property
    def scale(self):
        """return scale"""
        return self._scale

    @property
    def step_flat(self):
        """return step_flat"""
        return self._step_flat

    @staticmethod
    def zero():
        return learning_rate(step_heat=1000, scale=0)

    @staticmethod
    def from_0_to_1(heat, coef_heat):
        return learning_rate(heat, coef_heat, 10**10, 1)

    @staticmethod
    def from_1_to_0(burnin, coef_burnin):
        return learning_rate(0, 1, burnin, coef_burnin)

    def __call__(self, iter: int) -> float:
        if iter < self._step_flat:  # ensures zero value before growth
            return 0

        if iter < self._step_heat:  # before exp(coeff *(1-iter/step_heat))
            return self._scale * np.exp(
                self._coef_heat * (1 - float(iter) / self._step_heat)
            )

        if self._step_burnin is not None:
            if iter >= 1 + self._step_burnin:  # after (iter - step_burnin)^-coef
                return self._scale * pow(iter - self._step_burnin, -self._coef_burnin)

        return self._scale

    def __repr__(self) -> str:
        scale_msg = str(self._scale) + "*"
        if self._scale == 1:
            scale_msg = ""
        out = (
            self.__class__.__name__
            + " :"
            + "\n\t i -> | "
            + scale_msg
            + "exp("
            + str(self._coef_heat)
            + "*(1-i/"
            + str(self._step_heat)
            + "))\t if i < "
            + str(self._step_heat)
            + "\n\t      | "
            + scale_msg
            + "( i - "
            + str(self._coef_burnin)
            + ")^"
            + str(-self._coef_burnin)
            + "\t if i >= "
            + str(self._step_burnin)
            + "\n\t      | "
            + str(self._scale)
            + "\t otherwise"
        )

        return out

    def plot(self, label=None):
        import matplotlib.pyplot as plt

        if self.step_burnin is None:
            x = np.linspace(0, 2 * self.step_heat, num=4 * self.step_heat)
        else:
            x = np.linspace(0, 2 * self.step_burnin, num=4 * self.step_burnin)

        y = [self.__call__(i) for i in x]

        # print(x)
        # print(y)
        if label is None:
            return plt.plot(x, y)
        return plt.plot(x, y, label=label)


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

    f = learning_rate.from_1_to_0(20, 0.75)
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
