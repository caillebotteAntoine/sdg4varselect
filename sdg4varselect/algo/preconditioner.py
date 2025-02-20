"""
Module for preconditioners used in preconditioned stochastic gradient descent.

This module provides abstract and concrete implementations of preconditioners
to improve the convergence of stochastic gradient descent algorithms.

Created by antoine.caillebotte@inrae.fr
"""

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
    preconditioner : jnp.ndarray
        Last value computed of the preconditionner
    """

    def __init__(self) -> None:
        self._preconditioner = None

    @property
    def value(self) -> jnp.ndarray:
        """Return the last value computed of the preconditionner

        Returns
        -------
        jnp.ndarray
            last preconditionner"""
        return self._preconditioner

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


@jit
def compute_fisher(gradient, jac, jac_current, step_size_approx_sto, step_size_fisher):
    """Compute the preconditioned gradient using Fisher information.

    Parameters
    ----------
    gradient : jnp.ndarray
        The gradient to be preconditioned.
    jac : jnp.ndarray
        The last approximated jacobian.
    jac_current : jnp.ndarray
        The current jacobian.
    step_size_approx_sto : callable
        Function to compute the approximate step size.
    step_size_fisher : callable
        Function to compute the Fisher step size.

    Returns
    -------
    tuple
        A tuple containing:
            - jnp.ndarray: The precondition matrix.
            - jnp.ndarray: The preconditioned gradient.
            - jnp.ndarray: The approximated jacobian.
    """

    # Jacobian approximate
    jac = (1 - step_size_approx_sto) * jac + step_size_approx_sto * jac_current

    # gradient = self._jac.mean(axis=0)

    # Fisher computation
    fim = jac.T @ jac / jac.shape[0]

    preconditioner = step_size_fisher * fim + (1 - step_size_fisher) * jnp.eye(
        fim.shape[0]
    )
    grad_precond = jnp.linalg.solve(preconditioner, gradient)

    return preconditioner, grad_precond, jac


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
        AbstractPreconditioner.__init__(self)

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
        self._preconditioner = self._jac.T @ self._jac

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
        self._preconditioner, grad_precond, self._jac = compute_fisher(
            gradient, self._jac, jacobian, step_size_approx_sto, step_size_fisher
        )

        return grad_precond


@jit
def compute_adagrad(adagrad, gradient, regularization) -> jnp.ndarray:
    """Compute the preconditioned gradient using AdaGrad method.

    Parameters
    ----------
    adagrad : jnp.ndarray
        The value of the Adagrad all ready calculated.
    gradient : jnp.ndarray
        The gradient to be preconditioned.
    regularization : float, optional
        Regularization term to avoid division by zero (default is 1).

    Returns
    -------
    tuple
        A tuple containing:
            - jnp.ndarray: The precondition matrix.
            - jnp.ndarray: The preconditioned gradient.
            - jnp.ndarray: The approximated jacobian.
    """
    adagrad += gradient**2
    preconditioner = jnp.sqrt(regularization + adagrad)
    # assert preconditioner.shape == gradient.shape

    grad_precond = gradient / preconditioner
    return jnp.diag(preconditioner), grad_precond, adagrad


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
        AbstractPreconditioner.__init__(self)
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
        self._preconditioner = self._regularization * jnp.ones(shape=(jac_shape[1],))

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
        jnp.ndarray
            The preconditioned gradient.
        """
        self._preconditioner, grad_precond, self._adagrad = compute_adagrad(
            self._adagrad, gradient, self._regularization
        )
        self._adagrad_past.append(self._adagrad)
        return grad_precond


class Identity(AbstractPreconditioner):
    """Default preconditioner for stochastic gradient descent.

    This preconditioner is simply the identity
    """

    def __init__(self) -> None:
        AbstractPreconditioner.__init__(self)

    def initialize(self, jac_shape):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        """
        self._preconditioner = jnp.diag(jnp.ones(shape=(jac_shape[1],)))

    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute the preconditioned gradient using Identity information.

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
        jnp.ndarray
            The preconditioned gradient.
        """

        return gradient
