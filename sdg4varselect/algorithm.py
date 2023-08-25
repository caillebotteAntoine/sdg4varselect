import time
from collections import namedtuple
from functools import wraps

import jax.numpy as jnp

from sdg4varselect.learning_rate import learning_rate
from sdg4varselect.miscellaneous import default_arg, step_message
from sdg4varselect.solver import Solver

res_grad_tupletype = namedtuple(
    "res_grad_tupletype",
    (
        "theta",
        "jac",
        "fisher_info",
        "fisher_info_shrink",
        "grad",
        "grad_precond",
        "likelihood",
        "theta_diff",
    ),
)


def list_res_to_res_list(res, parametrization):
    res = [[res[i][j] for i in range(len(res))] for j in range(len(res[0]))]

    theta = [parametrization.reals1d_to_params(theta) for theta in res[0]]
    theta_jnp = jnp.array(
        [
            jnp.hstack([theta[i][j] for j in range(len(theta[i]))])
            for i in range(len(theta))
        ]
    )

    jac = res[1]
    fisher_info = res[2]
    fisher_info_shrink = res[3]
    grad = jnp.array(res[4])
    grad_precond = jnp.array(res[5])
    likelihood = jnp.array(res[6])
    theta_diff = jnp.array(res[7])

    return res_grad_tupletype(
        theta_jnp,
        jac,
        fisher_info,
        fisher_info_shrink,
        grad,
        grad_precond,
        likelihood,
        theta_diff,
    )


@default_arg
def algorithm_func(func=None, algorithm_name="algo"):
    """decorator to create algorithms inside a class"""

    @wraps(func)
    def new_algorithm(self, *args, **kwargs):
        if not self.is_init():
            raise AssertionError("parameters in solver are not initialized")
        self.start_solver(algorithm_name)
        out = func(self, *args, **kwargs)
        self.stop_solver(algorithm_name)

        return out

    return new_algorithm


class Algorithm(Solver):
    """An algorithm is a solver that uses specific algorithm and add a iter variables"""

    def __init__(self, key):
        """Constructor of algorithm."""
        super().__init__()

        self.__step_size = learning_rate()

        self.__iter = 0
        self.__start = 0
        self.__elapsed_time = 0

        self.__key = key

        self._verbatim = True

    @property
    def verbatim(self):
        """return verbatim"""
        return self._verbatim

    @verbatim.setter
    def verbatim(self, x):
        self._verbatim = x

    @property
    def iter(self):
        """return iter"""
        return self.__iter

    @property
    def step_size(self):
        """return the step_size of the algorithm"""
        return self.__step_size

    @step_size.setter
    def step_size(self, fct: learning_rate):
        if not isinstance(fct, learning_rate):
            raise TypeError("fct must be a burnin fct")
        self.__step_size = fct

    def __repr__(self) -> str:
        msg = super().__repr__()
        msg += f"\n\t*elapsed time = {self.__elapsed_time:8.3f}s"
        return msg

    # ===== step regardless of algo ===== #
    def gibbs_sampler_step(self, niter: int):
        """Compute niter steps of the gibbs sampler on latents variables"""

        for _ in range(niter):
            for var_lat in self.latent_variables.values():
                self.__key = var_lat.gibbs_sampler_step(
                    self.__key, self._theta_reals1d, **self._likelihood_kwargs
                )

    # ==== handling step method ===== #
    def step_message(self, niter) -> None:
        """increment the iteration count and display an intermediate message"""
        self.__iter += 1
        if self._verbatim:
            print(step_message(self.__iter, niter), end="\r")

    def reset_solver(self) -> None:
        """resets the resolution time and the number of iterations made for the solver"""
        self.__elapsed_time = 0
        self.__iter = 0
        for var_lat in self.latent_variables.values():
            var_lat.reset()

    def start_solver(self, name: str) -> None:
        """starts the resolution timer"""
        if self._verbatim:
            print(name + " started ! ")
        self.__start = time.time()

    def stop_solver(self, name: str) -> None:
        """stops the resolution timer"""
        end = time.time()
        self.__elapsed_time += end - self.__start
        if self._verbatim:
            print("\nend of the " + name + " !")

    #########################
    # ===== algorithm ===== #
    #########################

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm_func(algorithm_name="Heating")
    def heating(self, niter: int) -> None:
        """apply gibbs sampler several times in order to do a heating step
        before launching an algorithm"""
        for _ in range(niter):
            self.step_message(niter)
            # Simulation
            self.gibbs_sampler_step(1)


if __name__ == "__main__":
    pass
