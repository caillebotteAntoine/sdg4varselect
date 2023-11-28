import jax.numpy as jnp
import jax.random as jrd
from jax import jit

import itertools

from sdg4varselect.algorithm import (
    algorithm_func,
    Algorithm,
    learning_rate,
    res_grad_tupletype,
    list_res_to_res_list,
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


@jit
def gradient_descent_fisher_preconditionner(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    step_size_jac: float,
    step_size_fisher: float,
):
    """Compute one step of a gradient with perconditionner"""
    # Jacobian approximate
    jac = (1 - step_size_jac) * jac + step_size_jac * jac_current

    # Gradient
    grad = jac.mean(axis=0)

    fim = jac.T @ jac / jac.shape[0]
    fim = step_size_fisher * fim + (1 - step_size_fisher) * jnp.eye(fim.shape[0])

    grad_precond = jnp.linalg.solve(fim, grad)

    return jac, fim, grad, grad_precond


@jit
def gradient_descent_fisher_preconditionner_with_mask(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    # fisher_identity_mixture: bool,
    step_size_jac: float,
    step_size_fisher: float,
    fisher_mask: jnp.ndarray,
):
    """Compute one step of a gradient with perconditionner

    J_S = [J, O]
                  | J.T@J  0 |
    J_S.T @ J_S = | 0      0 |
                                          | J.T@J/N  0   |
    FIM = J_S.T @ J_S/N + diag(not mask) =| 0        I_p |

    """
    # Jacobian approximate
    jac = (1 - step_size_jac) * jac + step_size_jac * jac_current
    jac_shrink = jnp.where(fisher_mask, jac, 0)

    # Gradient
    grad = jac.mean(axis=0)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # for i in range(3, 5):
    #     grad = grad.at[i].set(0)

    # grad = grad.at[6].set(0)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    grad_shrink = jnp.where(fisher_mask, grad, 0)

    # ' jnp.where(jnp.array([True, False]), jnp.array([[1,2],[3,4]]),0)
    # '  = Array([[1, 0], [3, 0]])
    # '
    fim = jac_shrink.T @ jac_shrink / jac_shrink.shape[0] + jnp.diag(
        jnp.where(fisher_mask, 0, 1)
    )
    fim = step_size_fisher * fim + (1 - step_size_fisher) * jnp.eye(fim.shape[0])
    pred_grad_shrink = jnp.linalg.solve(fim, grad_shrink)

    grad_precond = jnp.where(fisher_mask, pred_grad_shrink, grad)

    return jac_shrink, fim, grad, grad_precond


class Gradient(Algorithm):
    def __init__(self, key, parametrization, params0):
        """Constructor of Gradient algorithm."""
        super().__init__(key)

        self.__step_size_fisher = learning_rate()
        self.__step_size_grad = learning_rate()

        self.parametrization = parametrization
        self._params0 = params0
        self.theta_reals1d = self._params0

    def reset_solver(self) -> None:
        """resets the resolution time and the number of iterations made for the solver"""
        super().reset_solver()
        self.theta_reals1d = self._params0

    @property
    def step_size_fisher(self):
        """return the step_size for fisher approximation"""
        return self.__step_size_fisher

    @step_size_fisher.setter
    def step_size_fisher(self, fct: learning_rate):
        if not isinstance(fct, learning_rate):
            raise TypeError("fct must be a burnin fct")
        self.__step_size_fisher = fct

    @property
    def step_size_grad(self):
        """return the size_grad for the step_size"""
        return self.__step_size_grad

    @step_size_grad.setter
    def step_size_grad(self, fct: learning_rate):
        if not isinstance(fct, learning_rate):
            raise TypeError("fct must be a burnin fct")
        self.__step_size_grad = fct

    # ===== step regardless of algo ===== #
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

    #########################
    # ===== algorithm ===== #
    #########################

    # = = = = = = = = = = = = = = = = = = = = = = = = = #
    @algorithm_func(algorithm_name="PSGD")
    def stochastic_gradient(
        self,
        niter: int,
        jac_likelihood,
        fisher_mask: jnp.ndarray,
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

        HD_mask = jnp.arange(len(self.theta_reals1d)) >= len(self.theta_reals1d) - p

        # c'est pas un iterator
        def GD(jac, grad):
            for _ in itertools.count():
                self.step_message(niter)

                # Simulation
                self.gibbs_sampler_step(gibbs_step)
                # Gradient descent
                jac_current = jac_likelihood(
                    self._theta_reals1d, **self._likelihood_kwargs
                )

                if jnp.any(jnp.isnan(jac_current)):
                    print("there is an nan in jac_current, stoping the algorithm !")
                    yield -1
                    break

                (
                    jac,
                    fisher_info,
                    grad,
                    grad_precond,
                ) = gradient_descent_fisher_preconditionner_with_mask(
                    jac,
                    jac_current,
                    fisher_mask=fisher_mask,
                    step_size_jac=self.step_size(self.iter),
                    step_size_fisher=self.step_size_fisher(self.iter),
                )

                self._theta_reals1d += self.step_size_grad(self.iter) * grad_precond

                if proximal_operator:
                    self._theta_reals1d = self.proximal_operator(
                        self.step_size_grad(self.iter),
                        prox_regul,
                        alpha=1,
                        HD_mask=HD_mask,
                    )

                isnan = {
                    "grad": jnp.any(jnp.isnan(grad)),
                    "jac": jnp.any(jnp.isnan(jac)),
                    "grad_precond": jnp.any(jnp.isnan(grad_precond)),
                    "fisher_info": jnp.any(jnp.isnan(fisher_info)),
                    "theta_reals1d": jnp.any(jnp.isnan(self._theta_reals1d)),
                }

                if True in isnan.values():
                    print(
                        f"there is an nan in {[key for x, key in enumerate(isnan) if x ]}, stoping the algorithm !"
                    )
                    yield -1
                    break

                yield (
                    self._theta_reals1d,
                    jac,
                    fisher_info,
                    grad,
                    grad_precond,
                    self.likelihood(self._theta_reals1d, **self._likelihood_kwargs),
                )

        res = list(
            itertools.islice(
                # itertools.takewhile(
                #     lambda x: not x.end_heating
                #     or (
                #         abs(x.grad_diff.sum()) > 10e-7
                #         and abs(x.theta_diff.sum()) > 10e-7
                #     ),
                GD(jac, grad),
                # ),
                niter,
            )
        )
        return list_res_to_res_list(res, self.parametrization)


def set_gradient_run_parameters(
    solver,
    activate_fim=False,
    activate_jac_approx=True,
    lr=1e-8,
    # Grad
    plateau_grad=400,
    scale_grad=1,
    plateau_grad_size=100,
    # Jac
    plateau_jac=300,
    plateau_jac_size=50,
    scale_jac=0.5,
    # Fim
    plateau_fim=750,
    plateau_fim_size=50,
    scale_fim=0.95,
):
    if activate_jac_approx:
        solver.step_size = learning_rate(
            plateau_jac,
            float(jnp.log(lr)),
            plateau_jac + plateau_jac_size,
            0.65,
            scale=scale_jac,
        )
    else:
        solver.step_size = learning_rate.one()

    if activate_fim:
        solver.step_size_fisher = learning_rate(
            plateau_fim,
            float(jnp.log(lr)),
            plateau_fim + plateau_fim_size,
            0.65,
            # step_flat=plateau_jac + 100,
            scale=scale_fim,
        )
    else:
        solver.step_size_fisher = learning_rate.zero()

    solver.step_size_grad = learning_rate(
        plateau_grad,
        float(jnp.log(lr)),
        plateau_grad + plateau_grad_size,
        0.65,
        scale=scale_grad,
        # step_flat=100,
    )

    # solver.step_size = solver.step_size_fisher
    # solver.step_size_grad = solver.step_size_fisher
    # learning_rate.from_1_to_0(plateau_fim + 100, 0.65)
    return solver


def get_random_params0(prng_key, params0, error=0.2, uniform_on=None):
    p = params0.copy()
    for key in p:
        key_new, prng_key = jrd.split(prng_key, 2)
        p[key] *= float(jrd.uniform(key_new, minval=1.0 - error))

    if uniform_on is not None:
        for key in uniform_on:
            key_new, prng_key = jrd.split(prng_key, 2)
            p[uniform_on] = jrd.uniform(
                prng_key, shape=p[uniform_on].shape, minval=-1, maxval=1
            )

    return p, key_new


if __name__ == "__main__":
    pass
