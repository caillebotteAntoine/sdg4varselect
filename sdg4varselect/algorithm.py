from sdg4varselect.solver import solver
from sdg4varselect.burnin_fct import burnin_fct
from sdg4varselect.parameter import par_grad
import time
from sdg4varselect.miscellaneous import step_message, default_arg


class algorithm(solver):
    def __init__(self):
        super().__init__()

        self.__step_size = burnin_fct()

        self.__iter = 0
        self.__start = 0
        self.__elapsed_time = 0

    def step_size(self, fct: burnin_fct):
        if not isinstance(fct, burnin_fct):
            raise TypeError("fct must be a burnin fct")
        self.__step_size = fct

    def to_csv(self, file_name: str) -> None:
        from sdg4varselect.csv_melter import solver2csv

        solver2csv(
            self.__elapsed_time,
            self.parameters,
            self.latent_variables,
            self.__step_size,
            self.__iter,
            file_name,
        )

    def __repr__(self) -> str:
        msg = super().__repr__()
        msg += "\n\t*elapsed time = {:8.3f}s".format(self.__elapsed_time)
        return msg

    # ===== step regardless of algo ===== #
    def stochastic_approximation(self, step_size: float) -> None:
        for par in self.parameters.values():
            par.step_stochastic_approximation(self, step_size)

    def gibbs_sampler_step(self, loglikelihood, niter: int) -> None:
        """Simulation step"""
        for i in range(niter):
            for var_lat in self.latent_variables.values():
                var_lat.gibbs_sampler_step(loglikelihood, self._theta, **self._data)

    def gradient_descent(self, loss_grad, step_size: float):
        grad_eval = loss_grad(self.theta_to_params(), **self._data)

        for par in self.parameters:
            self.parameters[par].step_gradient_descent(
                getattr(grad_eval, par), step_size
            )

    def gradient_descent_par(self, gradient: par_grad, step_size: float):
        gradient.step_size_stochastic_approximation = step_size
        gradient.step_stochastic_approximation(self, 1)

        for par in self._theta:
            self.parameters[par].step_gradient_descent(gradient[par], step_size)

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
            if not self.is_init():
                raise AssertionError("parameters in solver are not initialized")
            self.start_solver(algorithm_name)
            func(self, *args, **kwargs)
            self.stop_solver(algorithm_name)

        return new_algorithm

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm(algorithm_name="Heating")
    def Heating(self, niter: int, loglikelihood) -> None:

        for i in range(niter):
            self.__iter += 1
            step_message(self.__iter, niter)
            # Simulation
            self.gibbs_sampler_step(loglikelihood, 1)

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm(algorithm_name="SAEM")
    def SAEM(self, niter: int, loglikelihood, MCMC_step: int = 1) -> None:

        for i in range(niter):
            self.__iter += 1
            step_message(self.__iter, niter)
            # Simulation
            self.gibbs_sampler_step(loglikelihood, MCMC_step)
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
        print(self.__iter)

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


if __name__ == "__main__":
    from sdg4varselect import solver_init

    s = solver_init(
        algorithm(),
        theta0={"mu1": 5, "omega2_1": 0.5},
        mean_name="mu",
        variance_name="omega2_",
        mcmc_name="Z",
        dim={"Z1": 10},
        sd={"Z1": 0.5},
    )

    s.init_parameters()

    s.step_size(burnin_fct.from_1_to_0(140, 0.75))
    s.SAEM(150, loglikelihood=lambda theta: 0)
    print(s)
