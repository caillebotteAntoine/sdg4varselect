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
    freezed_components : jnp.ndarray
        Components of the frozen parameters.
    """

    def __init__(self) -> None:
        self._preconditioner = None
        self._freezed_components = None

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
    def initialize(self, jac_shape, freezed_components):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        freezed_components : jnp.ndarray
            The components of the frozen parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError


@jit
def compute_fisher(
    gradient,
    jac,
    jac_current,
    freezed_components,
    *,
    step_size_approx_sto,
    step_size_fisher,
):  # pylint: disable= too-many-arguments, too-many-locals
    """Compute the preconditioned gradient using Fisher information.

    Parameters
    ----------
    gradient : jnp.ndarray
        The gradient to be preconditioned.
    jac : jnp.ndarray
        The last approximated jacobian.
    jac_current : jnp.ndarray
        The current jacobian.
    freezed_components : jnp.ndarray
        The components of the frozen parameters.
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
    fim = jac.T @ jac / jac.shape[0] + jnp.diag(freezed_components)

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

    def initialize(self, jac_shape, freezed_components):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        freezed_components : jnp.ndarray
            The components of the frozen parameters.
        """
        self._jac = jnp.zeros(shape=jac_shape)  # approximated jac
        self._preconditioner = self._jac.T @ self._jac
        self._freezed_components = freezed_components

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
            gradient,
            self._jac,
            jacobian,
            self._freezed_components,
            step_size_approx_sto=step_size_approx_sto,
            step_size_fisher=step_size_fisher,
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
    preconditioner = jnp.sqrt(adagrad) + regularization
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

    def __init__(self, scale=None, regularization=1e-3) -> None:
        AbstractPreconditioner.__init__(self)
        self._adagrad = jnp.zeros(shape=(1, 1))
        self._past = []
        self._regularization = regularization
        self._scale = scale

    def initialize(self, jac_shape, freezed_components):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        freezed_components : jnp.ndarray
            The components of the frozen parameters.
        """
        if self._scale is not None:
            assert self._scale.shape == (jac_shape[1],)
        else:
            self._scale = jnp.ones(shape=(jac_shape[1],))

        self._adagrad = jnp.zeros(shape=(jac_shape[1],))
        self._past = [self._adagrad]
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
        self._past.append(self._adagrad)
        self._preconditioner *= self._scale
        return self._scale * grad_precond


class Identity(AbstractPreconditioner):
    """Default preconditioner for stochastic gradient descent.

    This preconditioner is simply the identity
    """

    def __init__(self, scale=None) -> None:
        AbstractPreconditioner.__init__(self)
        self._scale = scale

    def initialize(self, jac_shape, freezed_components):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        freezed_components : jnp.ndarray
            The components of the frozen parameters.
        """
        if self._scale is not None:
            assert self._scale.shape == (jac_shape[1],)
        else:
            self._scale = jnp.ones(shape=(jac_shape[1],))

        self._preconditioner = jnp.diag(self._scale)

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

        return self._scale * gradient


@jit
def _ema(x_past, x_new, gamma) -> jnp.ndarray:
    """Compute exponential moving average.

    gamma * x_past + (1-gamma)*x_new

    Parameters
    ----------
    x_past : jnp.ndarray
        past values.
    x_new : jnp.ndarray
        new values.
    gamma : float
        decay rates of average.

    Returns
    -------
    exponential moving average : jnp.ndarray
    """
    return gamma * x_past + (1 - gamma) * x_new


@jit
def compute_rmsprop(average_squared_gradients, gradient, regularization) -> jnp.ndarray:
    """Compute the preconditioned gradient using rmsprop method.

    Parameters
    ----------
    average_squared_gradients : jnp.ndarray
        exponentially weighted average of squared gradients
    gradient : jnp.ndarray
        The gradient to be preconditioned.
    regularization : float
        Regularization term to avoid division by zero.

    Returns
    -------
    tuple
        A tuple containing:
            - jnp.ndarray: The precondition matrix.
            - jnp.ndarray: The preconditioned gradient.
            - jnp.ndarray: sum of square of past gradients.
    """
    preconditioner = jnp.sqrt(average_squared_gradients) + regularization
    # assert preconditioner.shape == gradient.shape

    grad_precond = gradient / preconditioner
    return jnp.diag(preconditioner), grad_precond


class RMSP(AbstractPreconditioner):
    """RMSP preconditioner for stochastic gradient descent.

    This preconditioner adapts the learning rate based on the accumulated
    squared gradients.

    Parameters
    ----------
    regularization : float, optional
        Regularization term to avoid division by zero (default is 1).
    beta_1 : float
        decay rates of average of squared gradients.
    """

    def __init__(self, scale=None, regularization=1e-3, beta_1=0.9) -> None:
        AbstractPreconditioner.__init__(self)
        self._average_gradients2 = jnp.zeros(shape=(1, 1))
        self._regularization = regularization
        self._beta_1 = beta_1

        self._scale = scale
        self._past = []

    def initialize(self, jac_shape, freezed_components):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        freezed_components : jnp.ndarray
            The components of the frozen parameters.
        """
        if self._scale is not None:
            assert self._scale.shape == (jac_shape[1],)
        else:
            self._scale = jnp.ones(shape=(jac_shape[1],))

        self._average_gradients2 = jnp.zeros(shape=(jac_shape[1],))
        self._past = [self._average_gradients2]
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
        self._average_gradients2 = _ema(
            self._average_gradients2, gradient**2, self._beta_1
        )

        self._preconditioner, grad_precond = compute_rmsprop(
            average_squared_gradients=self._average_gradients2,
            gradient=gradient,
            regularization=self._regularization,
        )
        self._past.append(self._average_gradients2)
        self._preconditioner *= self._scale
        return self._scale * grad_precond


class ADAM(RMSP):
    """ADAM preconditioner for stochastic gradient descent.

    This preconditioner adapts the learning rate based on the accumulated
    squared gradients.

    Parameters
    ----------
    regularization : float, optional
        Regularization term to avoid division by zero (default is 1).
    beta_2 : float
        decay rates of average of squared gradients.
    beta_1 : float
        decay rates of average of gradients.
    """

    def __init__(
        self, scale=None, regularization=1e-8, beta_1=0.9, beta_2=0.999
    ) -> None:
        RMSP.__init__(self, scale=scale, regularization=regularization, beta_1=beta_1)
        self._average_gradients = jnp.zeros(shape=(1, 1))
        self._beta_2 = beta_2

    def initialize(self, jac_shape, freezed_components):
        """Initialize the Fisher preconditioner.

        Parameters
        ----------
        jac_shape : tuple
            The shape of the Jacobian matrix to initialize.
        freezed_components : jnp.ndarray
            The components of the frozen parameters.
        """
        if self._scale is not None:
            assert self._scale.shape == (jac_shape[1],)
        else:
            self._scale = jnp.ones(shape=(jac_shape[1],))

        RMSP.initialize(self, jac_shape, freezed_components)
        self._average_gradients = jnp.zeros(shape=(jac_shape[1],))

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
        self._average_gradients2 = _ema(
            self._average_gradients2, gradient**2, self._beta_2
        )
        self._average_gradients = _ema(self._average_gradients, gradient, self._beta_1)

        m_hat = self._average_gradients / (1 - self._beta_1 ** (step + 1))
        v_hat = self._average_gradients2 / (1 - self._beta_2 ** (step + 1))

        self._preconditioner, grad_precond = compute_rmsprop(
            average_squared_gradients=v_hat,
            gradient=m_hat,
            regularization=self._regularization,
        )

        self._preconditioner *= self._scale

        self._past.append(jnp.diag(self._preconditioner))
        return self._scale * grad_precond
