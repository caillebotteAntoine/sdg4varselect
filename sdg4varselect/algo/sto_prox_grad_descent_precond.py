# pylint: disable = duplicate-code, similar-code
"""
Module for Stochastic Proximal Gradient Descent algorithm preconditioned by the Fisher information matrix or another.

This module implements a Stochastic Proximal Gradient Descent (SPGD) algorithm,
where optimization is preconditioned using the Fisher information matrix.
The algorithm includes a proximal operator with elastic net regularization for model parameter
estimation and selection.
"""

from typing import Optional

from jax import jit
import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_high_dim_model import AbstractHDModel
from sdg4varselect.algo.preconditioner import AbstractPreconditioner
from sdg4varselect.outputs import Sdg4vsResults
from sdg4varselect.algo.sto_grad_descent_precond import (
    StochasticGradientDescentPrecond as SGD_Prec,
    BASE_PARAMETERS_GD,
)
from sdg4varselect._doc_tools import inherit_docstring


@jit
def _prox(
    theta: jnp.ndarray, stepsize: jnp.ndarray, lbd: jnp.ndarray, alpha: float = 1
) -> jnp.ndarray:
    """Apply the proximal operator with elastic net penalty to a parameter array.

    prox{stepsize, pen(lambda, alpha)} = argmin_theta (pen(theta') + 1/(2stepsize) ||theta-theta'||^2)

    Parameters
    ----------
    theta : jnp.ndarray
        Parameter array to which the proximal operator is applied.
    stepsize : jnp.ndarray
        Step size for the gradient update.
    lbd : jnp.ndarray
        Elastic net regularization parameter.
    alpha : jnp.ndarray, optional
        Elastic net mixing parameter between L1 (lasso) and L2 (ridge) regularization.
        Default is 1, corresponding to lasso regularization.

    Returns
    -------
    jnp.ndarray
        Updated parameter array after applying the proximal operator.
    """
    # stepsize = 1/adagrad
    id_shrink_too_big = theta >= stepsize * lbd * alpha
    id_shrink_too_litte = theta <= -stepsize * lbd * alpha

    return (
        id_shrink_too_big * (theta - stepsize * lbd * alpha)
        + id_shrink_too_litte * (theta + stepsize * lbd * alpha)
    ) / (
        1 + stepsize * lbd * (1 - alpha)  # alpha = 1 => res = 1
    )


@jit
def proximal_operator(
    theta_reals1d: jnp.ndarray,
    stepsize: jnp.ndarray,
    lbd: jnp.ndarray,
    alpha: float = 1,
    hd_mask=None,
) -> jnp.ndarray:
    """Apply the proximal operator to parameters using a mask.

    Parameters
    ----------
    theta_reals1d : jnp.ndarray
        1-dimensional array of real-valued model parameters.
    stepsize : float
        Step size for the gradient update.
    lbd : float
        Elastic net regularization parameter.
    alpha : float, optional
        Elastic net mixing parameter between L1 (lasso) and L2 (ridge) regularization.
        Default is 1, corresponding to lasso regularization.
    hd_mask : jnp.ndarray, optional
        Boolean mask indicating elements for applying the proximal operator.

    Returns
    -------
    jnp.ndarray
        Array of parameters after applying the proximal operator.
    """
    assert isinstance(lbd, (float, jnp.ndarray))
    if isinstance(lbd, jnp.ndarray):
        assert len(lbd.shape) == 0 or lbd.shape == theta_reals1d.shape

    return jnp.where(
        hd_mask,
        _prox(theta_reals1d, stepsize, lbd, alpha),
        theta_reals1d,
    )


@inherit_docstring
class StochasticProximalGradientDescentPrecond(SGD_Prec):
    __doc__ = f"""Stochastic Proximal Gradient Descent algorithm with Fisher matrix preconditioning.

    This class implements a preconditioned SPGD algorithm using the Fisher information matrix
    and supports elastic net regularization through a proximal operator.

    Parameters
    ----------
    {BASE_PARAMETERS_GD}

    lbd : float
        Elastic net regularization parameter.
    alpha : float, optional
        Elastic net mixing parameter between L1 (lasso) and L2 (ridge) regularization.
        Default is 1, corresponding to lasso regularization.

    Attributes
    ----------
    hd_mask : jnp.ndarray
        Mask indicating high-dimensional parameters
    """

    def __init__(
        self,
        preconditioner: AbstractPreconditioner,
        lbd: Optional[float] = None,
        alpha: Optional[float] = 1.0,
        **kwargs,
    ):
        SGD_Prec.__init__(self, preconditioner, **kwargs)

        self._lbd = lbd
        self._alpha = alpha

        # initial algo parameter
        self._hd_mask = None

    @property
    def lbd(self) -> float:
        """Get the elastic net regularization parameter.

        Returns
        -------
        float
            Elastic net regularization parameter.
        """
        return self._lbd

    @lbd.setter
    def lbd(self, lbd: float):
        self._lbd = lbd

    @property
    def hd_mask(self) -> jnp.ndarray:
        """Get the mask indicating high-dimensional parameters.

        Returns
        -------
        jnp.ndarray
            Mask indicating high-dimensional parameters.
        """
        return self._hd_mask

    @hd_mask.setter
    def hd_mask(self, mask: jnp.ndarray):
        self._hd_mask = mask

    def results_warper(  # pylint: disable = missing-return-doc
        self, model, theta0_reals1d, data, results
    ) -> Sdg4vsResults:
        out = SGD_Prec.results_warper(self, model, theta0_reals1d, data, results)
        out.update_bic(model)
        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel] = None,
        theta_reals1d: jnp.ndarray = None,
        freezed_components: jnp.ndarray = None,
        log_likelihood_kwargs: dict = None,
    ) -> None:
        if isinstance(model, AbstractHDModel):
            self.hd_mask = model.hd_mask

        assert self._hd_mask is not None, " HD mask must be initialized!"
        assert (
            self._hd_mask.shape == theta_reals1d.shape
        ), "HD mask must have the same shape as the theta array"

        # self._hd_mask = (
        #     jnp.arange(theta_reals1d.shape[0]) >= model.P
        # )  # jnp.ar#model.hd_mask
        # self._fisher_mask = jnp.invert(self._hd_mask)
        SGD_Prec._initialize_algo(
            self, model, theta_reals1d, freezed_components, log_likelihood_kwargs
        )

    # ============================================================== #

    def _one_proximal_operator(self, theta_reals1d, step) -> jnp.ndarray:
        """Apply the proximal operator for a single step.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        step : int
            The current iteration step.

        Returns
        -------
        jnp.ndarray
            Parameter array after applying the proximal operator.
        """
        if self._lbd is None:
            return theta_reals1d
        # prox_{A, \lambda ||.||} ( \theta - A^{-1} grad)
        # adagrad : A = diag((sqrt(grad^2)+eps)/gamma_0)
        return proximal_operator(
            theta_reals1d,
            stepsize=self._step_size(step) / jnp.diag(self._preconditioner.value),
            lbd=self._lbd,
            alpha=self._alpha,
            hd_mask=self._hd_mask,
        )

    # ============================================================== #
    def _algorithm_one_step(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        # Simulation
        self._one_simulation(log_likelihood_kwargs, theta_reals1d)

        # Gradient descent
        (
            theta_reals1d,
            grad,
            grad_precond,
            preconditioner,
        ) = self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )

        # Proximal operator
        theta_reals1d = self._one_proximal_operator(theta_reals1d, step)

        # grad_log_likelihood_marginal = self.grad_log_likelihood_marginal(
        #     model, log_likelihood_kwargs, theta_reals1d, size=30
        # )
        grad_log_likelihood_marginal = None

        return (
            theta_reals1d,  # 0
            grad,  # 1
            grad_precond,  # 2
            preconditioner,  # 3
            grad_log_likelihood_marginal,  # 4
        )
