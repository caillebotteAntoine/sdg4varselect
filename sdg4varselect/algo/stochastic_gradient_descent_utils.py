"""
Module for jitted function used in stochastic gradient descent

Create by antoine.caillebotte@inrae.fr
"""

from jax import jit
import jax.numpy as jnp


@jit
def gradient_descent_fisher_preconditionner(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    # fisher_identity_mixture: bool,
    step_size_approx_sto: float,
    step_size_fisher: float,
):
    """Compute one step of a gradient with perconditionner"""
    # Jacobian approximate
    jac_approx = (1 - step_size_approx_sto) * jac + step_size_approx_sto * jac_current

    # Gradient
    grad = jac_approx.mean(axis=0)
    fim = jac_approx.T @ jac_approx / jac_approx.shape[0]

    fim = step_size_fisher * fim + (1 - step_size_fisher) * jnp.eye(fim.shape[0])
    grad_precond = jnp.linalg.solve(fim, grad)

    return jac_approx, fim, grad_precond


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
    hd_mask=None,
) -> jnp.ndarray:
    """apply the proximal operator on a mask"""
    return jnp.where(
        hd_mask,
        prox(theta_reals1d, stepsize, lbd, alpha),
        theta_reals1d,
    )
