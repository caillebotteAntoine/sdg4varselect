"""
Module for utiliy functions and chain class.

Create by antoine.caillebotte@inrae.fr
"""
from warnings import warn
import numpy as np


class Chain:
    """
    Represents a chain of values with history.

    Attributes:
    ----------
        _name (str): Name of the chain.
        _data (numpy.ndarray): Current value of the chain.
        _size (int): Size of the chain.
        _chain (list[numpy.ndarray]): List to store the history of the chain.

    Methods:
    ----------
        __init__(self, x0: float, size: int = 1, name: str = "NA"):
            Initializes a new Chain object with the given parameters.

        __repr__(self) -> str:
            Returns a string representation of the current value of the chain.

        print(self) -> str:
            Prints the current value of the chain along with its previous values.

        reset(self):
            Resets the chain to its initial state given when the object was created.

        __len__(self) -> int:
            Returns the size of the chain.

        update_chain(self):
            Appends a copy of the current value of the chain to the history.

    Properties:
    ----------
        data(self) -> numpy.ndarray:
            Returns the current value of the chain.

        chain(self) -> list[numpy.ndarray]:
            Returns the history of the chain.

        name(self) -> str:
            Returns the name of the chain.
    """

    def __init__(self, x0: float, size: int = 1, name="NA"):
        """Initializes a new Chain object with the given parameters.

        Parameters:
        ----------
            x0 (float): Initial value for the chain.
            size (int, optional): Size of the chain. Default is 1.
            name (str, optional): Name of the chain. Default is "NA".
        """
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
        prefix = "" if self._name == "NA" else "[" + self._name + "]"
        msg = np.array2string(self._data, prefix=prefix, suffix=")")
        return prefix + "(" + msg + ")"

    def print(self) -> str:
        """Prints the current value of the chain along with its previous values."""
        msg_add = "\n previous values = "
        msg_add += str(np.array(self._chain[1:])).replace("\n", ", ")

        msg = str(self) + msg_add
        print(msg)
        return msg

    def reset(self):
        """Resets the chain to its initial state given when the object was created."""
        x0 = self._chain[0][0]
        self._data = np.array([float(x0) for i in range(self._size)])
        self._chain = []
        self.update_chain()

    def __len__(self) -> int:
        return self._size

    def update_chain(self):
        """append to the chain a copy of the current value of data"""
        self._chain.append(self._data.copy())

    @property
    def data(self) -> np.ndarray:
        """Returns the current value of the chain."""
        return self._data

    @property
    def chain(self) -> list[np.ndarray]:
        """Returns the history of the chain."""
        return self._chain

    @property
    def name(self) -> str:
        """Returns the name of the chain."""
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
    os: str, progress: float, maxbar: int = 50, indicator: str = ">", **_kwargs
) -> str:
    """
    Generates a progress bar string based on the given parameters.

    Parameters:
        os (str): The initial string to which the progress bar is appended.
        progress (float): The progress value (between 0 and 1).
        maxbar (int, optional): The maximum length of the progress bar. Default is 50.
        indicator (str, optional): The indicator character at the end of the progress bar. Default is ">".
        **kwargs: Additional keyword arguments.

    Returns:
        str: The string with the appended progress bar.

    Example:
        loadbar("Loading: ", 0.6, 20, "*")
        # "Loading: [===========*        ]"
    """
    nbar = int(progress * maxbar)
    if nbar > maxbar:
        nbar = maxbar

    os += "["
    for _ in range(nbar):
        os += "="
    os += indicator
    for _ in range(nbar, maxbar):
        os += " "
    os += "]"
    return os


def loadnumber(
    os: str, number: int, bigest_number: int, unit: str = "", **_kwargs
) -> str:
    """
    Adds a formatted representation of a load number of type n/N to the given string.

    Parameters:
        os (str): The initial string to which the number representation is appended.
        number (int): The current number.
        bigest_number (int): The largest number for formatting.
        unit (str, optional): The unit to be appended to the number. Default is an empty string.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The string with the appended number representation.

    Example:
        loadnumber("Progress: ", 25, 100, "%")
        # "Progress: 25/100%"
    """
    number_str = str(number)
    max_str = str(bigest_number)

    if len(number_str) < len(max_str):
        diff_len = len(max_str) - len(number_str)
        for _ in range(diff_len):
            os += " "

    os += number_str + "/" + max_str + unit
    return os


def step_message(iteration: int, max_iter: int, **kwargs) -> str:
    """
    Generates a step message with a progress bar and load number representation.
    you should use `print(os, end="\r") with it.

    Parameters:
        iteration (int): The current iteration.
        max_iter (int): The maximum number of iterations.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The generated step message.

    Example:
        result = step_message(3, 10)
        # " 3/10 [=====>              ]"
    """
    os = ""
    os = loadnumber(os, iteration, max_iter, **kwargs) + " "
    os = loadbar(os, float(iteration) / max_iter, **kwargs)

    return os


if __name__ == "__main__":
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
