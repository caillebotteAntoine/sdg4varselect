"""
Module for preconditioners used in preconditioned stochastic gradient descent.

This module provides abstract and concrete implementations of preconditioners
to improve the convergence of stochastic gradient descent algorithms.

Created by antoine.caillebotte@inrae.fr
"""

import functools
from abc import ABC, abstractmethod
from copy import deepcopy


from jax import jit
import jax.numpy as jnp

from sdg4varselect.learning_rate import default_step_size


class AbstractPreconditioner(ABC):
    """
    Abstract base class for preconditioners in stochastic gradient descent.

    This class defines the interface for preconditioners that can modify the
    gradient to improve convergence rates in optimization problems.

    Parameters
    ----------
    *args : tuple
        Positional arguments for initialization.
    **kwargs : dict
        Keyword arguments for initialization.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute the preconditioned gradient.

        Parameters
        ----------
        gradient : jnp.ndarray
            The gradient to be preconditioned.
        jacobian : jnp.ndarray
            The Jacobian matrix used for preconditioning.
        step : float
            The current step size value.

        Returns
        -------
        jnp.ndarray
            The preconditioned gradient.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(self, jac_shape):
        """Initialize the preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class Fisher(AbstractPreconditioner):
    """Fisher preconditioner for stochastic gradient descent.

    This preconditioner uses the Fisher information matrix to modify the
    gradient in order to improve convergence properties.

    Parameters
    ----------
    step_size_approx_sto : callable
        Function to compute the approximate step size.
    step_size_fisher : callable
        Function to compute the Fisher step size.
    """

    def __init__(self) -> None:
        self._jac = jnp.zeros(shape=(1, 1))

        self._step_size_approx_sto = deepcopy(default_step_size)
        self._step_size_approx_sto.heating.step = None

        self._step_size_fisher = deepcopy(self._step_size_approx_sto)
        self._step_size_fisher.max = 0.9

    def initialize(self, jac_shape):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        """
        self._jac = jnp.zeros(shape=jac_shape)  # approximated jac

    @functools.partial(jit, static_argnums=0)
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute the preconditioned gradient using Fisher information.

        Parameters
        ----------
        gradient : jnp.ndarray
            The gradient to be preconditioned.
        jacobian : jnp.ndarray
            The Jacobian matrix used for preconditioning.
        step : float
            The current step size.

        Returns
        -------
        tuple
            A tuple containing:
                - jnp.ndarray: The precondition matrix.
                - jnp.ndarray: The preconditioned gradient.
        """
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
    """AdaGrad preconditioner for stochastic gradient descent.

    This preconditioner adapts the learning rate based on the accumulated
    squared gradients.

    Parameters
    ----------
    regularization : float, optional
        Regularization term to avoid division by zero (default is 1).
    """

    def __init__(self, regularization=1) -> None:
        self._adagrad = jnp.zeros(shape=(1, 1))
        self._adagrad_past = []
        self._regularization = regularization

    def initialize(self, jac_shape):
        """Initialize the AdaGrad preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        """
        self._adagrad = jnp.zeros(shape=(jac_shape[1],))
        self._adagrad_past = [self._adagrad]

    @functools.partial(jit, static_argnums=0)
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute the preconditioned gradient using AdaGrad.

        Parameters
        ----------
        gradient : jnp.ndarray
            The gradient to be preconditioned.
        jacobian : jnp.ndarray
            The Jacobian matrix used for preconditioning.
        step : float
            The current step size.

        Returns
        -------
        tuple
            A tuple containing:
                - jnp.ndarray: The precondition matrix.
                - jnp.ndarray: The preconditioned gradient.
        """
        self._adagrad += gradient**2
        self._adagrad_past.append(self._adagrad)

        precond = jnp.sqrt(self._regularization + self._adagrad)
        assert precond.shape == gradient.shape

        grad_precond = gradient / precond
        return jnp.diag(precond), grad_precond


class Identity(AbstractPreconditioner):
    """Default preconditioner for stochastic gradient descent.

    This preconditioner is simply the identity
    """

    def __init__(self) -> None:
        pass

    def initialize(self, jac_shape):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        """

    @functools.partial(jit, static_argnums=0)
    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute the preconditioned gradient using Fisher information.

        Parameters
        ----------
        gradient : jnp.ndarray
            The gradient to be preconditioned.
        jacobian : jnp.ndarray
            The Jacobian matrix used for preconditioning.
        step : float
            The current step size.

        Returns
        -------
        tuple
            A tuple containing:
                - jnp.ndarray: The precondition matrix.
                - jnp.ndarray: The preconditioned gradient.
        """

        return jnp.diag(jnp.ones(gradient.shape)), gradient
