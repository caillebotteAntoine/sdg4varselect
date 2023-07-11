import itertools
import time
from collections import namedtuple
from functools import wraps


import jax.numpy as jnp
from jax import jit
import jax

from sdg4varselect.learning_rate import learning_rate
from sdg4varselect.miscellaneous import default_arg, step_message
from sdg4varselect.solver import Solver

from functools import partial

res_grad_tupletype = namedtuple(
    "res_grad_tupletype",
    (
        "theta",
        "jac",
        "fisher_info",
        "grad",
        "grad_precond",
        "likelihood",
        "step_size",
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
    grad = jnp.array(res[3])
    grad_precond = jnp.array(res[4])
    likelihood = jnp.array(res[5])
    step_size = jnp.array(res[6])

    return res_grad_tupletype(
        theta_jnp, jac, fisher_info, grad, grad_precond, likelihood, step_size
    )


@jit
def prox(
    theta: jnp.ndarray, stepsize: float, lbd: float, alpha: float = 1  # shape = (p,)
) -> jnp.ndarray:
    """apply the proximal operator on params for the elastic net panlty
    prox{stepsize, pen(lambda, alpha)} = argmin_theta (pen(theta') + 1/(2stepsize) ||theta-theta'||^2)

    gamma step size
    alpha choice between ridge and lasso
    lambda tuning parameter of elastic net
    """
    id_shrink_too_big = theta >= stepsize * lbd * alpha
    id_shrink_too_litte = theta <= -stepsize * lbd * alpha

    return (
        id_shrink_too_big * (theta - stepsize * lbd * alpha)
        + id_shrink_too_litte * (theta + stepsize * lbd * alpha)
    ) / (
        1 + stepsize * lbd * (1 - alpha)  # alpha = 1 => res = 1
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


# @partial(jit, static_argnums=(4,))
@jit
def gradient_descent_fisher_preconditionner_with_mask(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    fisher_identity_mixture: bool,
    step_size: float,
    fisher_mask: jnp.ndarray,
):
    """compute one step of a gradient with perconditionner"""
    grad = jac_current.mean(axis=0)

    # Jacobian approximate
    jac = (1 - step_size) * jac + step_size * jac_current

    # id_mask = jnp.nonzero(fisher_mask)[0]
    # id_rest = jnp.nonzero(1 - fisher_mask)[0]

    # jac_shrink = jac[:, id_mask]
    jac_shrink = jnp.where(fisher_mask, jac, 0)
    fisher_info_shrink = jax.lax.cond(
        fisher_identity_mixture,
        lambda fim: step_size * fim + (1 - step_size) * jnp.eye(fim.shape[0]),
        lambda fim: fim,
        jac_shrink.T @ jac_shrink / jac_shrink.shape[0],
    ) + jnp.diag(jnp.where(fisher_mask, 0, 1))

    grad_shrink = jnp.linalg.solve(
        fisher_info_shrink, jnp.where(fisher_mask, grad, 0)
    )  # grad[id_mask])

    theta_step = step_size * jnp.where(
        fisher_mask, grad_shrink, grad
    )  # jnp.hstack([grad_shrink, grad[id_rest]])

    return jac, fisher_info_shrink, step_size * grad, theta_step


@jit
def gradient_descent_fisher_preconditionner(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    fisher_identity_mixture: bool,
    step_size: float,
):
    """compute one step of a gradient with perconditionner"""
    grad = jac_current.mean(axis=0)

    # Jacobian approximate
    jac = (1 - step_size) * jac + step_size * jac_current

    fisher_info = jax.lax.cond(
        fisher_identity_mixture,
        lambda fim: step_size * fim + (1 - step_size) * jnp.eye(fim.shape[0]),
        lambda fim: fim,
        jac.T @ jac / jac.shape[0],
    )

    theta_step = step_size * jnp.linalg.solve(fisher_info, grad)

    return jac, fisher_info, grad, theta_step


@jit
def gradient_descent(
    jac_current: jnp.ndarray,
    step_size: float,
):
    """compute one step of a gradient"""
    grad = jac_current.mean(axis=0)
    grad /= jnp.sqrt((grad**2).sum())
    return grad, step_size * grad


class Algorithm(Solver):
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

    def stochastic_approximation(self, old, new, step_size):
        """compute a stochastic approximation on self.__theta_reals1d with theta_new"""
        return (1 - step_size) * old + step_size * new

    def proximal_operator(
        self, stepsize: float, lbd: float, alpha: float = 1, HD_mask=None, p=None
    ):
        if HD_mask is None:
            assert p is not None
            HD_mask = jnp.arange(len(self.theta_reals1d)) >= len(self.theta_reals1d) - p

        # theta = (mu_remap, gamma_remap, beta_grand_dim_remap) in R^d
        # theta[3:] -> appliqué prox
        return jnp.where(
            HD_mask,
            prox(self.theta_reals1d, stepsize, lbd, alpha),
            self.theta_reals1d,
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

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm_func(algorithm_name="PSGD")
    def stochastic_gradient(
        self,
        niter: int,
        jac_likelihood,
        fisher_preconditionner: bool,
        fisher_mask: jnp.ndarray,
        smart_start: int,
        proximal_operator: bool,
        prox_regul: float = 1.0,
        p: int = 0,
        gibbs_step: int = 1,
        *args,
        **kwargs,
    ):
        """Stochastic gradient"""

        N, theta_size = jac_likelihood(
            self._theta_reals1d, **self._likelihood_kwargs
        ).shape

        jac = jnp.zeros((N, theta_size))
        grad = jnp.zeros(theta_size)
        fisher_info = jnp.zeros((theta_size, theta_size))

        HD_mask = jnp.arange(len(self.theta_reals1d)) >= len(self.theta_reals1d) - p

        # c'est pas un iterator
        def GD(jac, grad, fisher_info):
            yield res_grad_tupletype(
                self._theta_reals1d,
                jac,
                jac.T @ jac,
                grad,
                grad,
                self.likelihood(self._theta_reals1d, **self._likelihood_kwargs),
                0,
            )

            for _ in itertools.count():
                self.step_message(niter)
                factor = self.__step_size(self.__iter)

                # Simulation
                self.gibbs_sampler_step(gibbs_step)
                # Gradient descent
                if jnp.any(jnp.isnan(grad)):
                    break

                jac_current = jac_likelihood(
                    self._theta_reals1d, **self._likelihood_kwargs
                )

                if fisher_preconditionner:
                    (
                        jac,
                        fisher_info,
                        grad,
                        theta_step,
                    ) = gradient_descent_fisher_preconditionner_with_mask(
                        jac,
                        jac_current,
                        fisher_identity_mixture=self.__iter < smart_start,
                        fisher_mask=fisher_mask,
                        step_size=factor,
                    )
                else:
                    grad, theta_step = gradient_descent(
                        jac_current,
                        step_size=factor,
                    )

                self._theta_reals1d += theta_step

                if proximal_operator:
                    # theta = (mu_remap, gamma_remap, beta_grand_dim_remap) in R^d
                    # theta[3:] -> appliqué prox
                    self._theta_reals1d = self.proximal_operator(
                        factor, prox_regul, HD_mask=HD_mask
                    )

                yield res_grad_tupletype(
                    self._theta_reals1d,
                    jac,
                    fisher_info,
                    grad,
                    theta_step,
                    self.likelihood(self._theta_reals1d, **self._likelihood_kwargs),
                    factor,
                )

        res = list(itertools.islice(GD(jac, grad, fisher_info), niter))
        return list_res_to_res_list(res, self.parametrization)


if __name__ == "__main__":
    pass
