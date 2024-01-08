import numpy as np
import jax.numpy as jnp

from warnings import warn


class Chain:
    def __init__(self, x0: float, size: int = 1, name="NA"):
        """Constructor of chain."""
        self._name = name

        if isinstance(x0, (list, np.ndarray)):
            if len(x0) > 1:
                warn(
                    "an array has been given as initial value, only the first component is taken into account"
                )

            x0 = x0[0]

        if not isinstance(x0, (int, float, np.floating, np.integer)):
            raise TypeError("x0 must be an integer or a float")

        if not isinstance(size, (int, float, np.floating, np.integer)):
            raise TypeError("size must be an integer or a float")

        self._data = np.array([float(x0) for i in range(size)])
        self._size = size

        self._chain: list[np.ndarray] = []
        self.update_chain()  # append x0 to the chain

    def __repr__(self) -> str:
        prefix = self._type
        if self._name != "NA":
            prefix += "[" + self._name + "]"
        prefix += "("
        msg = np.array2string(self._data, prefix=prefix, suffix=")")
        return prefix + msg + ")"

    def print(self) -> str:
        msg_add = "\n previous values = "
        msg_add += str(np.array(self._chain)).replace("\n", ", ")

        msg = str(self) + msg_add
        print(msg)
        return msg

    def reset(self):
        x0 = self._chain[0][0]
        self._data = np.array([float(x0) for i in range(self._size)])
        self._chain = []
        self.update_chain()

    # at 25/12
    # def init(self, x0):
    #     for i in range(self._size):
    #         self._data[i] = x0

    def __len__(self) -> int:
        return self._size

    def update_chain(self):
        """append to the chain a copy of the current value of data"""
        self._chain.append(self._data.copy())

    @property
    def data(self) -> np.ndarray:
        """returns the current value of data"""
        return self._data

    @property
    def chain(self) -> list[np.ndarray]:
        """returns the data chain"""
        return self._chain

    @property
    def name(self) -> str:
        """returns chain's name"""
        return self._name


if __name__ == "__main__":
    x = Chain(0, 3)
    print(x)

    y = x.data
    print(y)

    y[0] = 2

    print(y)
    print(x)


def loadbar(
    os: str, progress: float, maxbar: int = 50, indicator: str = ">", **kwargs
) -> str:
    nbar = int(progress * maxbar)
    if nbar > maxbar:
        nbar = maxbar

    os += "["
    for i in range(nbar):
        os += "="
    os += indicator
    for i in range(nbar, maxbar):
        os += " "
    os += "]"
    return os


def loadnumber(os: str, number: int, max: int, unit: str = "", **kwargs) -> str:
    number_str = str(number)
    max_str = str(max)

    if len(number_str) < len(max_str):
        diff_len = len(max_str) - len(number_str)
        for i in range(diff_len):
            os += " "

    os += number_str + "/" + max_str + unit
    return os


def step_message(iter: int, max_iter: int, **kwargs) -> str:
    os = ""
    os = loadnumber(os, iter, max_iter, **kwargs) + " "
    os = loadbar(os, float(iter) / max_iter, **kwargs)
    # if iter == max_iter - 1:
    #     os += "\n"
    return os  # you should use `print(os, end="\r")`


if __name__ == "__main__":
    import numpy as np

    for i in range(1000):
        print(step_message(i, 1000), end="\r")
    print("HELLO WORLD")

    # 25/12
    # x = np.arange(0, 1e6)

    # def np_sum(x):
    #     np.sum(x)

    # def mysum(x):
    #     res = 0
    #     for v in x:
    #         res += v
    #     return res

    # difftime(np_sum, sum, mysum)(x)

    # print([time2string(x) for x in [2.5, 0.6, 2.3e-4, 2.3e-3, 2.2e-7, 2.2e-6, 2.1e-10]])
