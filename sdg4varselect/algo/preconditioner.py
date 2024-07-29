"""
Module for preconditioner used in preconditioned stochastic gradient descent

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0115, C0116


import functools
from jax import jit
import jax.numpy as jnp

from abc import abstractmethod
from sdg4varselect.learning_rate import create_multi_step_size


class AbstractPreconditioner:

    def __init__(self, jac_shape, settings) -> None:
        pass

    @abstractmethod
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """
        Return preconditioned_gradient
        """

    @abstractmethod
    def initialize(self, jac_shape):
        """
        Initialize the preconditioner
        """


class Fisher(AbstractPreconditioner):

    def __init__(self, settings) -> None:
        self._jac = jnp.zeros(shape=(1, 1))

        step_sizes = create_multi_step_size(list(settings), num_step_size=2)
        (
            self._step_size_approx_sto,
            self._step_size_fisher,
        ) = (
            step_sizes[0],
            step_sizes[1],
        )

    def initialize(self, jac_shape):
        self._jac = jnp.zeros(shape=jac_shape)  # approximated jac

    @functools.partial(jit, static_argnums=0)
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        step_size_approx_sto = self._step_size_approx_sto(step)
        step_size_fisher = self._step_size_fisher(step)

        # Jacobian approximate
        self._jac = (
            1 - step_size_approx_sto
        ) * self._jac + step_size_approx_sto * jacobian

        # gradient = self._jac.mean(axis=0)

        # Fisher computation
        fim = self._jac.T @ self._jac / self._jac.shape[0]

        precond = step_size_fisher * fim + (1 - step_size_fisher) * jnp.eye(
            fim.shape[0]
        )
        grad_precond = jnp.linalg.solve(precond, gradient)

        return precond, grad_precond


class AdaGrad(AbstractPreconditioner):

    def __init__(self, regularization=1) -> None:
        self._adagrad = jnp.zeros(shape=(1, 1))
        self._adagrad_past = []
        self._regularization = regularization

    def initialize(self, jac_shape):
        self._adagrad = jnp.zeros(shape=(jac_shape[1],))
        self._adagrad_past = [self._adagrad]

    @functools.partial(jit, static_argnums=0)
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:

        self._adagrad += gradient**2
        self._adagrad_past.append(self._adagrad)

        precond = 1 / jnp.sqrt(self._regularization + self._adagrad)
        assert precond.shape == gradient.shape

        grad_precond = precond * gradient
        return precond, grad_precond


class FisherAdaGrad(AbstractPreconditioner):

    def __init__(self, P, settings, regularization=1) -> None:

        self._fisher = Fisher(settings)
        self._adagrad = AdaGrad(regularization=regularization)

        self._p = P
        self._hd_mask = jnp.zeros(1)

    def initialize(self, jac_shape):
        self._fisher.initialize(jac_shape)
        self._adagrad.initialize(jac_shape)

        self._hd_mask = jnp.arange(jac_shape[1]) < self._p

    @functools.partial(jit, static_argnums=0)
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute one step of a gradient with perconditionner

        J_S = [J, O]
                      | J.T@J  0 |
        J_S.T @ J_S = | 0      0 |
                                                     | (J.T@J/N)^-1  0   |
        precond = (J_S.T @ J_S/N)^-1 + diag(ada)  =  | 0             ada |
        """
        jacobian_shrink = jnp.where(self._hd_mask, jacobian, 0)
        grad_shrink = jnp.where(self._hd_mask, gradient, 0)

        precond_fisher, grad_precond_fisher = self._fisher.get_preconditioned_gradient(
            grad_shrink, jacobian_shrink, step
        )
        precond_adagrad, grad_precond_adagrad = (
            self._adagrad.get_preconditioned_gradient(gradient, jacobian, step)
        )

        precond = precond_fisher + 1 / precond_adagrad
        grad_precond = jnp.where(
            self._hd_mask, grad_precond_fisher, grad_precond_adagrad
        )

        return precond, grad_precond
