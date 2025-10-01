"""Module for Chain class.

Create by antoine.caillebotte@inrae.fr
"""

from warnings import warn
import numpy as np


class Chain:
    """
    Represents a chain of values with history.

    Attributes
    ----------
        _name (str): Name of the chain.
        _data (numpy.ndarray): Current value of the chain.
        _size (int): Size of the chain.
        _chain (list[numpy.ndarray]): List to store the history of the chain.

    Methods
    -------
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

    Properties
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

        Parameters
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
        """return the representation string of the chain"""
        prefix = "" if self._name == "NA" else "[" + self._name + "]"
        msg = np.array2string(self._data, prefix=prefix, suffix=")")
        return prefix + "(" + msg + ")"

    def print(self) -> str:
        """Prints the current value of the chain along with its previous values.
        :return: return printed message"""
        msg_add = "\n previous values = "
        msg_add += str(np.array(self._chain[1:])).replace("\n", ", ")

        msg = str(self) + msg_add
        print(msg)
        return msg

    def reset(self, x0: float = None) -> None:
        """Resets the chain to its initial state given when the object was created.

        Parameters
        ----------
            x0 : float, optional
                If None, x0 will be equal to the initial value of the chain.
        """
        if x0 is None:
            x0 = self._chain[0][0]

        for i in range(self._size):
            self._data[i] = x0

        self._chain = []
        self.update_chain()

    def __len__(self) -> int:
        """length of the chain
        :return: size of the chain"""
        return self._size

    def update_chain(self):
        """append to the chain a copy of the current value of data"""
        self._chain.append(self._data.copy())

    @property
    def data(self) -> np.ndarray:
        """Current value of the chain.
        :return: current value of the chain"""
        return self._data

    @property
    def chain(self) -> list[np.ndarray]:
        """History of the chain.
        :return: history list _chain"""
        return self._chain

    @property
    def name(self) -> str:
        """Returns the name of the chain.
        :return: the str _name"""
        return self._name


if __name__ == "__main__":
    x = Chain(0, 3)
    print(x)

    y = x.data
    print(y)

    y[0] = 2

    print(y)
    print(x)
