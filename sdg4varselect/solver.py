import numpy as np
from warnings import warn
from typing import Optional
import time

# from chain import chain
from MCMC import MCMC_chain
from burnin_fct import burnin_fct
from parameter import parameter, par_grad

from csv_melter import solver2csv

from miscellaneous import step_message, default_arg


class solver:
    def __init__(self):
        """Constructor of solver."""
        self.__parameters = {}
        self.__theta = {}
        self.__latent_variables = {}
        self.__global_variables = {}

        self.__data = {}

        self.__step_size = burnin_fct()
        self.__iter = 0
        self.__start = 0
        self.__elapsed_time = 0

    def theta(self):
        return self.__theta

    def data(self):
        return self.__data

    def to_csv(self, file_name: str) -> None:
        solver2csv(
            self.__elapsed_time,
            self.__parameters,
            self.__latent_variables,
            self.__step_size,
            self.__iter,
            file_name,
        )

    def get_global(self, name):
        if name in self.__global_variables:
            return self.__global_variables[name]
        raise KeyError(name + " isn't in global variables")

    def step_size(self, fct: burnin_fct):
        if not isinstance(fct, burnin_fct):
            raise TypeError("fct must be a burnin fct")
        self.__step_size = fct

    def set_data(self, *name_args) -> None:
        if len(name_args) > 1:
            for n in name_args:
                self.set_data(n)
        else:
            name = name_args[0]
            if isinstance(name, str):
                if name in self.__parameters:
                    self.__data[name] = self.__parameters[name].data()
                elif name in self.__latent_variables:
                    self.__data[name] = self.__latent_variables[name].data()
                elif name in self.__global_variables:
                    self.__data[name] = self.__global_variables[name]
                else:
                    raise KeyError(
                        name
                        + "does not exist neiter in parameters or in latent variables or in global variables"
                    )
            else:
                raise TypeError("name must be a str or a list of str")

    def add_variable(self, name: str, x) -> None:
        self.__global_variables[name] = x

    def add_parameter(self, x: parameter) -> None:
        name = x.name()
        if name == "NA":
            raise ValueError("parameters must have a name to be added to the solver")

        self.__parameters[name] = x
        self.__theta[name] = x.data()

    def init_parameters(self) -> None:
        for par in self.__parameters.values():

            name = par.linked_name
            if name in self.__latent_variables:
                var = self.__latent_variables[name].data()
            elif name in self.__global_variables:
                var = self.__global_variables[name]
            else:
                raise KeyError(
                    name
                    + "does not exist neiter in latent variables or in global variables"
                )

            par.init(var)

        from collections import namedtuple

        theta_namedtuple = namedtuple(
            "theta_namedtuple", [x for x in self.__parameters]
        )
        self.__theta = theta_namedtuple(**self.__theta)

    def parametrization(self, **kwargs):
        import parametrization_cookbook.jax as pc

        sorted_kwargs = {}
        for par in self.__parameters:
            if par not in kwargs:
                raise ValueError(par + " is missing in the parametrization tuple")
            else:
                sorted_kwargs[par] = kwargs[par]

        return pc.NamedTuple(**sorted_kwargs)

    def add_MCMC(
        self,
        name: str,
        x0: float,
        size: int,
        sd: float,
        mean_name: str,
        variance_name: Optional[str] = None,
    ) -> None:

        if not isinstance(mean_name, str):
            raise TypeError("mean_name must be a str")
        if mean_name not in self.__parameters:
            raise KeyError(mean_name + " does not exist in parameters")
        mean = self.__parameters[mean_name].data()

        variance = np.array([1])
        if variance_name is None:
            warn(
                "no variance was provided, the default variance was set to 1 as hyper parameter"
            )
        else:
            if not isinstance(variance_name, str):
                raise TypeError("variance_name must be a str")
            if variance_name not in self.__parameters:
                raise KeyError(variance_name + "does not exist in parameters")

            variance = self.__parameters[variance_name].data()

        self.__latent_variables[name] = MCMC_chain(x0, size, sd, mean, variance, name)

    def __repr__(self) -> str:
        msg = "[ == solver === ]\n\t*latent variables :"
        for var in self.__latent_variables.values():
            msg += "\n\t\t-" + str(var)

        msg += "\n\t*parameters :"
        for par in self.__parameters.values():
            msg += "\n\t\t-" + str(par)

        # msg += "\n\t*theta :" + str(self.__theta)

        msg += "\n\t*data : "
        for k in self.__data:
            msg += k + ", "

        return msg + "\n\t*elapsed time = " + str(self.__elapsed_time) + "s"

    # ===== step regardless of algo ===== #
    def stochastic_approximation(self, step_size: float) -> None:
        for par in self.__parameters.values():
            par.step_stochastic_approximation(self, step_size)

    def gibbs_sampler_step(self, loglikelihood, niter: int) -> None:
        """Simulation step"""
        for i in range(niter):
            for var_lat in self.__latent_variables.values():
                var_lat.gibbs_sampler_step(loglikelihood, self.__theta, **self.__data)

    def gradient_descent(self, loss_grad, step_size: float):
        grad_eval = loss_grad(self.__theta, **self.__data)
        """
        print(grad_eval)
        print(type(grad_eval))
        print(self.__parameters)
        print(type(self.__parameters))

        print(grad_eval)
        print(self.__parameters)"""

        for par in self.__parameters:
            # print(getattr(grad_eval, par))
            # print(self.__parameters[par])
            self.__parameters[par].step_gradient_descent(
                getattr(grad_eval, par), step_size
            )

    def gradient_descent_par(self, gradient: par_grad, step_size: float):
        gradient.step_size_stochastic_approximation = step_size
        gradient.step_stochastic_approximation(self, 1)

        for par in self.__theta:
            self.__parameters[par].step_gradient_descent(gradient[par], step_size)

    # ==== handling step method ===== #
    def reset_solver(self) -> None:
        self.__elapsed_time = 0
        self.__iter = 0

    def start_solver(self, name: str) -> None:
        print(name + " started ! ")
        self.__start = time.time()

    def stop_solver(self, name: str) -> None:
        end = time.time()
        self.__elapsed_time += end - self.__start
        print("\nend of the " + name + " !")

    #########################
    # ===== algorithm ===== #
    #########################
    @default_arg
    def algorithm(func, algorithm_name="algo"):
        def new_algorithm(self, *args, **kwargs):
            self.start_solver(algorithm_name)
            func(self, *args, **kwargs)
            self.stop_solver(algorithm_name)

        return new_algorithm

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm(algorithm_name="SAEM")
    def SAEM(self, niter: int, loglikelihood, MCMC_step: int = 1) -> None:

        for i in range(niter):
            self.__iter += 1
            step_message(self.__iter, niter)
            # Simulation
            self.gibbs_sampler_step(loglikelihood, MCMC_step)
            # print(sum([loglikelihood(i, **self.__data) for i in range(N)]))
            # Stochastic approximation
            self.stochastic_approximation(self.__step_size(self.__iter))

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm(algorithm_name="SGD")
    def SGD(self, niter: int, loglikelihood, loss_grad, MCMC_step: int = 1) -> None:
        for i in range(niter):
            self.__iter += 1
            step_message(self.__iter, niter)
            # Simulation
            self.gibbs_sampler_step(loglikelihood, MCMC_step)
            # Gradient descent
            self.gradient_descent(loss_grad, self.__step_size(self.__iter))

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm(algorithm_name="SGD2")
    def SGD2(self, niter: int, loglikelihood, gradient, MCMC_step: int = 1) -> None:
        for i in range(niter):
            self.__iter += 1
            step_message(self.__iter, niter)
            # Simulation
            self.gibbs_sampler_step(loglikelihood, MCMC_step)
            # Gradient descent
            self.gradient_descent_par(gradient, self.__step_size(self.__iter))


def solver_init(s: solver, theta0, mean_name, variance_name, mcmc_name, dim, sd):
    from parameter import par_mean, par_variance

    for k in theta0:

        if mean_name in k:
            name_len = len(mean_name)

            mcmc = mcmc_name + k[name_len:]
            variance = variance_name + k[name_len:]

            s.add_parameter(par_mean(theta0[k], mcmc, k))

            if variance in theta0:
                s.add_parameter(par_variance(theta0[variance], mcmc, variance))
                s.add_MCMC(mcmc, theta0[k], dim[mcmc], sd[mcmc], k, variance)
            else:
                s.add_MCMC(mcmc, theta0[k], dim[mcmc], sd[mcmc], k)
    return s


if __name__ == "__main__":
    # === Data simulation === #
    from data_sim import loglikelihood, s

    s.step_size(burnin_fct.from_1_to_0(1450, 0.75))
    s.SAEM(1500, loglikelihood)
    print(s)

    s.to_csv("../cout.txt")
