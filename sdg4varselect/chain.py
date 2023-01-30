import numpy as np
from warnings import warn


class chain:
    used_name: list[str] = []

    def __init__(self, x0: float, size: int = 1, name=None, type="chain"):
        """Constructor of chain."""
        self.__type = type
        if name in chain.used_name:
            raise ValueError(name + " already used as chain name")
        if name is None:
            self.__name = "NA"
        else:
            self.__name = name
            chain.used_name.append(name)

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

        self.__chain: list[np.ndarray] = []
        self.update_chain()  # append x0 to the chain

    def __repr__(self) -> str:
        prefix = self.__type
        if self.__name != "NA":
            prefix += "[" + self.__name + "]"
        prefix += "("
        msg = np.array2string(self._data, prefix=prefix, suffix=")")
        return prefix + msg + ")"

    def print(self):
        msg_add = "\n previous values = "
        msg_add += str(np.array(self.__chain)).replace("\n", ", ")

        msg = str(self) + msg_add
        print(msg)
        return msg

    def init(self, x0):
        for i in range(self._size):
            self._data[i] = x0

    def __len__(self) -> int:
        return self._size

    def update_chain(self):
        self.__chain.append(self._data.copy())

    def data(self) -> np.ndarray:
        return self._data

    def chain(self) -> list[np.ndarray]:
        return self.__chain

    def name(self) -> str:
        return self.__name

    def type(self) -> str:
        return self.__type


if __name__ == "__main__":

    x = chain(0, 3)
    print(x)

    y = x.data()
    print(y)

    y[0] = 2

    print(y)
    print(x)
