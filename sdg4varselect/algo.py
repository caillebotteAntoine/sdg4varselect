from jax import jit
import jax.numpy as jnp
import numpy as np

import itertools
from collections import namedtuple

from sdg4varselect.data_handler import Data_handler
from sdg4varselect.learning_rate import create_multi_step_size
from sdg4varselect.miscellaneous import step_message


@jit
def gradient_descent_fisher_preconditionner_with_mask(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    # fisher_identity_mixture: bool,
    step_size_approx_sto: float,
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
    jac = (1 - step_size_approx_sto) * jac + step_size_approx_sto * jac_current
    jac_shrink = jnp.where(fisher_mask, jac, 0)

    # Gradient
    grad = jac.mean(axis=0)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # for i in range(0, 5):
    #     grad = grad.at[i].set(0)

    # for i in range(0, 6):
    #     grad = grad.at[i].set(0)

    # for i in range(10, 14):
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


def proximal_operator(
    theta_reals1d: jnp.ndarray,
    stepsize: float,
    lbd: float,
    alpha: float = 1,
    HD_mask=None,
):
    return jnp.where(
        HD_mask,
        prox(theta_reals1d, stepsize, lbd, alpha),
        theta_reals1d,
    )


class SPG_FIM:
    def __init__(self, PRNGKey, dh: Data_handler, settings, lbd=None, alpha=1.0):
        self._data_handler = dh

        self.PRNGKey = PRNGKey

        self._lbd = lbd
        self._alpha = alpha

        (
            self._step_size_grad,
            self._step_size_approx_sto,
            self._step_size_fisher,
        ) = create_multi_step_size(settings, num_step_size=3)

    @property
    def data(self):
        return self._data_handler.data

    @property
    def mcmc(self):
        return self._data_handler.latent_variables.values()

    estim_res = namedtuple("estim_res", ("theta", "FIM", "grad"))

    @classmethod
    def labelswitch(self, res_estim):
        res = [
            [res_estim[i][j] for i in range(len(res_estim))]
            for j in range(len(res_estim[0]))
        ]

        res = self.estim_res(
            theta=jnp.array(res[0]), FIM=res[2], grad=jnp.array(res[4])
        )
        return res

    # @functools.partial(jit, static_argnums=0)
    def simulation(self, theta_reals1d):
        # Simulation
        for var_lat in self._data_handler.latent_variables.values():
            self.PRNGKey = var_lat.gibbs_sampler_step(
                self.PRNGKey, theta_reals1d, **self.data
            )

    # @functools.partial(jit, static_argnums=(0, 1))
    def SPG_FIM_one_iter(
        self,
        jac_likelihood,
        theta_reals1d: jnp.ndarray,
        jac: jnp.ndarray,
        FIM_MASK,
        HD_MASK,
        step_size,
    ):
        # Simulation
        self.simulation(theta_reals1d)

        # Gradient descent
        jac_current = jac_likelihood(theta_reals1d, **self.data)

        if jnp.any(jnp.isnan(jac_current)):
            print("there is an nan in jac_current, stoping the algorithm !")
            return -1

        (
            jac,
            fisher_info,
            grad,
            grad_precond,
        ) = gradient_descent_fisher_preconditionner_with_mask(
            jac,
            jac_current,
            fisher_mask=FIM_MASK,
            step_size_approx_sto=step_size[1],
            step_size_fisher=step_size[2],
        )

        theta_reals1d += step_size[0] * grad_precond

        if self._lbd is not None:
            theta_reals1d = proximal_operator(
                theta_reals1d,
                step_size[0],
                lbd=self._lbd,
                alpha=self._alpha,
                HD_mask=HD_MASK,
            )

        return (
            theta_reals1d,
            jac,
            fisher_info,
            grad,
            grad_precond,
        )

    def algorithm(
        self,
        jac_likelihood,
        niter,
        theta_reals1d: jnp.ndarray,
        jac0,
        FIM_MASK,
        HD_MASK,
    ):
        jac = jac0
        iter = 0

        for _ in itertools.count():
            iter += 1
            print(step_message(iter, niter), end="\r")

            step_size = [
                self._step_size_grad(iter),
                self._step_size_approx_sto(iter),
                self._step_size_fisher(iter),
            ]

            (
                theta_reals1d,
                jac,
                fisher_info,
                grad,
                grad_precond,
            ) = self.SPG_FIM_one_iter(
                jac_likelihood,
                theta_reals1d,
                jac,
                FIM_MASK,
                HD_MASK,
                step_size,
            )

            isnan = {
                "grad": jnp.any(jnp.isnan(grad)),
                "jac": jnp.any(jnp.isnan(jac)),
                "grad_precond": jnp.any(jnp.isnan(grad_precond)),
                "fisher_info": jnp.any(jnp.isnan(fisher_info)),
                "theta_reals1d": jnp.any(jnp.isnan(theta_reals1d)),
            }

            if True in isnan.values():
                print(
                    f"there is an nan in {[key for x, key in enumerate(isnan) if x ]}, stoping the algorithm !"
                )
                yield -1
                break

            yield (
                theta_reals1d,
                jac,
                fisher_info,
                grad,
                grad_precond,
                # likelihood(theta_reals1d, **dh.data),
            )

    def fit(
        self,
        jac_likelihood,
        niter,
        DIM_HD,
        theta0_reals1d: jnp.ndarray,
    ):
        (DIM_THETA,) = theta0_reals1d.shape

        # mask for proximal operator
        HD_MASK = jnp.arange(DIM_THETA) >= DIM_THETA - DIM_HD
        # mask for fisher preconditionning
        FIM_MASK = np.invert(HD_MASK)

        # c'est pas un iterator

        jac_shape = jac_likelihood(theta0_reals1d, **self.data).shape
        return list(
            itertools.islice(
                self.algorithm(
                    jac_likelihood,
                    niter,
                    theta0_reals1d,
                    jac0=jnp.zeros(shape=jac_shape),
                    FIM_MASK=FIM_MASK,
                    HD_MASK=HD_MASK,
                ),
                niter,
            )
        )
