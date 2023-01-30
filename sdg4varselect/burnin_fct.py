import numpy as np


class burnin_fct:
    def __init__(
        self,
        step_heat: int = 0,
        coef_heat: float = 1,
        step_burnin: int = 0,
        coef_burnin: float = 1,
        scale: float = 1,
    ):

        if not isinstance(step_heat, int):
            raise TypeError("step_heat must be int")
        self.__step_heat = step_heat

        if not isinstance(coef_heat, (int, float)):
            raise TypeError("coef_heat must be int or float")
        self.__coef_heat = coef_heat

        if not isinstance(step_burnin, int):
            raise TypeError("step_burnin must be int")
        self.__step_burnin = step_burnin

        if not isinstance(coef_burnin, (int, float)):
            raise TypeError("coef_burnin must be int or float")
        self.__coef_burnin = coef_burnin

        if not isinstance(scale, (int, float)):
            raise TypeError("scale must be int or float")
        self.__scale = scale

    @staticmethod
    def from_0_to_1(heat, coef_heat):
        return burnin_fct(heat, coef_heat, 10**10, 1)

    @staticmethod
    def from_1_to_0(burnin, coef_burnin):
        return burnin_fct(0, 1, burnin, coef_burnin)

    def __call__(self, iter: int) -> float:

        if iter < self.__step_heat:  # before
            return self.__scale * np.exp(
                self.__coef_heat * (1 - float(iter) / self.__step_heat)
            )

        if iter > self.__step_burnin:  # after
            return self.__scale * pow(iter - self.__step_burnin, -self.__coef_burnin)

        return self.__scale


if __name__ == "__main__":

    f = burnin_fct(10, -4, 20, 0.75)
    x = np.arange(40)
    y = [f(i) for i in x]
    print(x)

    import matplotlib.pyplot as plt

    plt.step(x, y)
    plt.show()

    f = burnin_fct.from_0_to_1(10, -4)
    y = [f(i) for i in range(20)]

    f = burnin_fct.from_1_to_0(20, 0.75)
    y += [f(i) for i in range(20, 40)]

    plt.step(x, y)
    plt.show()

    print(type(f))

    print(isinstance(f, burnin_fct))
