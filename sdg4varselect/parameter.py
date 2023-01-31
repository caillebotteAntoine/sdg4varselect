from typing import Optional
import numpy as np
from sdg4varselect.chain import chain


class parameter(chain):
    def __init__(
        self,
        x0: float,
        size: int,
        name: Optional[str] = None,
        type: str = "parameter",
    ):
        """Constructor of parameter."""
        chain.__init__(self, x0, size, name, type)

    def init(self, x: np.ndarray):
        pass

    def step_gradient_descent(self, grad: np.ndarray, step_size: float):
        self._data += step_size * np.array(grad)
        self.update_chain()

    def step_stochastic_approximation(self, solver, step_size: float):
        self.estimate(solver)
        if step_size != 1:
            old_data = self._data.copy()  # memorize the past value

            self._data *= step_size
            self._data += (1 - step_size) * old_data

        self.update_chain()

    def estimate(self, solver) -> None:
        pass

    def type(self) -> str:
        return "parameter"


class linked_parameter(parameter):
    def __init__(
        self,
        x0: float,
        size: int,
        linked_name: str,
        name: Optional[str] = None,
        type: str = "parameter",
    ):
        """Constructor of parameter."""
        super().__init__(x0, size, name, type)

        self.linked_name = linked_name
        self._linked_chain = np.array([])

    def init(self, x: np.ndarray):
        # super().init(x0)
        self._linked_chain = x


class par_mean(linked_parameter):
    def __init__(self, x0: float, linked_name: str, name: Optional[str] = None):
        super().__init__(x0, 1, linked_name, name, "mean")

    def estimate(self, solver) -> None:
        self._data[0] = np.mean(self._linked_chain)


class par_variance(linked_parameter):
    def __init__(self, x0: float, linked_name: str, name: Optional[str] = None):
        super().__init__(x0, 1, linked_name, name, "variance")

    def estimate(self, solver) -> None:
        self._data[0] = np.var(self._linked_chain)


class par_noise_variance(linked_parameter):
    def __init__(
        self, x0: float, linked_name: str, non_linear_fct, name: Optional[str] = None
    ):
        super().__init__(x0, 1, linked_name, name, "noise's variance")
        self.__non_linear_fct = non_linear_fct

    def estimate(self, s) -> None:
        # from solver import solver

        pred = self.__non_linear_fct(**s.data())
        noise = self._linked_chain - pred

        self._data[0] = np.var(noise)


class par_grad_ind(parameter):
    def __init__(
        self,
        size: int,
        id: int,
        loss_grad_id,
        name: str = "gradient",
    ):
        super().__init__(0.0, size, name + str(id), "gradient")
        self.__loss_grad_id = loss_grad_id

        self.__id = id

    def estimate(self, s) -> None:
        # from solver import solver
        out = self.__loss_grad_id(s.theta(), self.__id, **s.data())
        out = np.concatenate([out[k] for k in out])

        for i in range(self._size):
            self._data[i] = out[i]


class par_grad(parameter):
    def __init__(
        self, theta: dict[str, np.ndarray], grad_ind: list[par_grad_ind], name: str
    ):

        super().__init__(0.0, len(grad_ind[0]), name + str(id), "gradient")
        self.__grad_ind = grad_ind

        self.step_size_stochastic_approximation = 1.0

        self.__index = {}
        i = 0
        for v in theta:
            self.__index[v] = [i + k for k in range(len(theta[v]))]
            i += len(theta[v])

        # print(self.__index)

    def compute_all_individual_grad(self, solver):
        for i in range(self._size):
            self._data[i] = 0.0

        for i in range(len(self.__grad_ind)):
            self.__grad_ind[i].step_stochastic_approximation(
                solver, self.step_size_stochastic_approximation
            )

            self._data += self.__grad_ind[i].data()

    def estimate(self, solver) -> None:
        self.compute_all_individual_grad(solver)
        self._data /= len(self.__grad_ind)

    def __getitem__(self, key):
        # print(key + " = " + str(self.__index[key]))
        return self._data[self.__index[key]]


if __name__ == "__main__":
    pass
