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

    def __init__(self, length_history=1) -> None:
        self._preconditioner = None
        self._freezed_components = None

        self._length_history = length_history
        self._values = []

    @property
    def freezed_components(self) -> jnp.ndarray:
        """Return the components of the frozen parameters.

        Returns
        -------
        jnp.ndarray
            Components of the frozen parameters.
        """
        return self._freezed_components

    @freezed_components.setter
    def freezed_components(self, value: jnp.ndarray):
        """Set the components of the frozen parameters.

        Parameters
        ----------
        value : jnp.ndarray
            Components of the frozen parameters.
        """
        self._freezed_components = value

    @property
    def value(self) -> jnp.ndarray:
        """Return the last value computed of the preconditionner

        Returns
        -------
        jnp.ndarray
            last preconditionner"""
        return self._preconditioner

    @property
    def history(self) -> list:
        """Return the history of preconditioner values.

        Returns
        -------
        list
            A list containing the history of preconditioner values.
        """
        return self._values

    def append_value_to_history(self):
        """Append a new value to the history of preconditioner values."""
        self._values.append(deepcopy(self._preconditioner))
        if len(self._values) > self._length_history:
            self._values.pop(0)

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

        """
        self._freezed_components = freezed_components


def compute_estimated_fisher(
    jac,
    jac_current,
    freezed_components,
    *,
    step_size_approx_sto,
    step_size_identity_mixture=0.0,
):
    """Compute the estimated Fisher information matrix.
    This function computes the Fisher information matrix using the Jacobian
    approximation and the current Jacobian. It combines the two using a step size
    for the approximation and a step size for the identity mixture.

    Parameters
    ----------
    jac : jnp.ndarray
        The last approximated Jacobian.
    jac_current : jnp.ndarray
        The current Jacobian.
    freezed_components : jnp.ndarray
        The components of the frozen parameters.
    step_size_approx_sto : float
        The step size for the Jacobian approximation.
    step_size_identity_mixture : float, optional
        The step size for the identity mixture (default is 0.0).

    Returns
    -------
    jnp.ndarray
        The estimated Fisher information matrix.
    """
    # Jacobian approximate
    jac = (1 - step_size_approx_sto) * jac + step_size_approx_sto * jac_current

    # Fisher computation
    fim = jac.T @ jac / jac.shape[0] + jnp.diag(freezed_components)

    # Identity mixture
    fim = (
        step_size_identity_mixture * jnp.eye(fim.shape[0])
        + (1 - step_size_identity_mixture) * fim
    )

    return fim


@jit
def compute_fisher(
    gradient,
    jac,
    jac_current,
    freezed_components,
    *,
    step_sizes,
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
    step_sizes : tuple
        A tuple containing the step sizes for the jac approximation and identity mixture.

    Returns
    -------
    tuple
        A tuple containing:
            - jnp.ndarray: The precondition matrix.
            - jnp.ndarray: The preconditioned gradient.
            - jnp.ndarray: The approximated jacobian.
    """

    preconditioner = compute_estimated_fisher(
        jac,
        jac_current,
        freezed_components,
        step_size_approx_sto=step_sizes[0],
        step_size_identity_mixture=1 - step_sizes[1],
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

    def __init__(self, length_history) -> None:
        AbstractPreconditioner.__init__(self, length_history=length_history)

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
        AbstractPreconditioner.initialize(self, jac_shape, freezed_components)

        self._jac = jnp.zeros(shape=jac_shape)  # approximated jac
        self._preconditioner = self._jac.T @ self._jac
        self._values = []

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
            step_sizes=[step_size_approx_sto, step_size_fisher],
        )
        self.append_value_to_history()

        return grad_precond


@jit
def compute_adagrad(
    adagrad, gradient, regularization, freezed_components
) -> jnp.ndarray:
    """Compute the preconditioned gradient using AdaGrad method.

    Parameters
    ----------
    adagrad : jnp.ndarray
        The value of the Adagrad all ready calculated.
    gradient : jnp.ndarray
        The gradient to be preconditioned.
    regularization : float, optional
        Regularization term to avoid division by zero (default is 1).
    freezed_components : jnp.ndarray
        The components of the frozen parameters.

    Returns
    -------
    tuple
        A tuple containing:
            - jnp.ndarray: The precondition matrix.
            - jnp.ndarray: The preconditioned gradient.
            - jnp.ndarray: The approximated jacobian.
    """
    adagrad += jnp.where(freezed_components, 0, gradient**2)
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
        AbstractPreconditioner.initialize(self, jac_shape, freezed_components)

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
            self._adagrad, gradient, self._regularization, self._freezed_components
        )
        self._past.append(self._adagrad)
        self._preconditioner /= self._scale
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


class AdagradFisher(AdaGrad):
    """AdagradFisher preconditioner for stochastic gradient descent.

    This preconditioner use Adagrad to adapt the
    learning rate based on the accumulated squared gradients and also estimate the Fisher
    information matrix.

    Parameters
    ----------
    regularization : float, optional
        Regularization term to avoid division by zero (default is 1).
    """

    def __init__(self, length_history, scale=None, regularization=1e-3) -> None:
        AdaGrad.__init__(self, scale=scale, regularization=regularization)
        self._length_history = length_history

    def get_preconditioned_gradient(self, gradient, jacobian, step) -> jnp.ndarray:
        """Compute the preconditioned gradient using AdagradFisher.

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

        self._preconditioner = compute_estimated_fisher(
            jacobian,
            jacobian,
            self._freezed_components,
            step_size_approx_sto=1,
            step_size_identity_mixture=0,
        )
        self.append_value_to_history()

        return AdaGrad.get_preconditioned_gradient(self, gradient, jacobian, step)


def preconditioner_factory(name, **kwargs):
    """
    Create a preconditioner instance by name.

    Parameters
    ----------
    name : str
        The name of the preconditioner ('fisher', 'adagrad', 'identity', 'rmsprop', 'adam', 'adagradfisher').
    **kwargs : dict
        Arguments passed to the preconditioner constructor.

    Returns
    -------
    AbstractPreconditioner
        An instance of the requested preconditioner.

    Raises
    ------
    ValueError
        If the preconditioner name is not recognized.
    """
    _preconditioners = {
        "fisher": Fisher,
        "adagrad": AdaGrad,
        "identity": Identity,
        "rmsprop": RMSP,
        "adam": ADAM,
        "adagradfisher": AdagradFisher,
    }
    key = name.lower()
    if key not in _preconditioners:
        raise ValueError(f"Unknown preconditioner: {name}")
    return _preconditioners[key](**kwargs)
