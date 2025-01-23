"""
Module for Stochastic Proximal Gradient Descent algorithm preconditioned by the Fisher information matrix or another.

This module implements a Stochastic Proximal Gradient Descent (SPGD) algorithm,
where optimization is preconditioned using the Fisher information matrix.
The algorithm includes a proximal operator with elastic net regularization for model parameter
estimation and selection.

Created by antoine.caillebotte@inrae.fr
"""

from datetime import datetime
from typing import Optional

from jax import jit
import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.models.abstract.abstract_high_dim_model import AbstractHDModel
from sdg4varselect.algo.preconditioner import AbstractPreconditioner

from sdg4varselect.algo.sto_grad_descent_precond import (
    StochasticGradientDescentPrecond as SGD_Prec,
)


@jit
def _prox(
    theta: jnp.ndarray, stepsize: float, lbd: float, alpha: float = 1
) -> jnp.ndarray:
    """Apply the proximal operator with elastic net penalty to a parameter array.

    prox{stepsize, pen(lambda, alpha)} = argmin_theta (pen(theta') + 1/(2stepsize) ||theta-theta'||^2)

    Parameters
    ----------
    theta : jnp.ndarray
        Parameter array to which the proximal operator is applied.
    stepsize : float
        Step size for the gradient update.
    lbd : float
        Elastic net regularization parameter.
    alpha : float, optional
        Elastic net mixing parameter between L1 (lasso) and L2 (ridge) regularization.
        Default is 1, corresponding to lasso regularization.

    Returns
    -------
    jnp.ndarray
        Updated parameter array after applying the proximal operator.
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


class StochasticProximalGradientDescentPrecond(SGD_Prec):
    """Stochastic Proximal Gradient Descent algorithm with Fisher matrix preconditioning.

    This class implements a preconditioned SPGD algorithm using the Fisher information matrix
    and supports elastic net regularization through a proximal operator.

    Parameters
    ----------
    prngkey : jax.random.PRNGKey
        A PRNG key, consumable by random functions.
    preconditioner : AbstractPreconditioner
        An instance of a preconditioner that can computes precondtionned gradient
    lbd : float
        Elastic net regularization parameter.
    alpha : float, optional
        Elastic net mixing parameter between L1 (lasso) and L2 (ridge) regularization.
        Default is 1, corresponding to lasso regularization.
    threshold : float, optional
        A threshold for the gradient magnitude to determine convergence. The default value is 1e-4.
    max_iter : int
        Maximum number of iterations allowed for the algorithm.
    ntry : int
        Number of attempts to retry the algorithm if an error is encountered.
    ntry_max : int
        Maximum number of retry attempts to reset `_ntry` after each algorithm run.
    partial_fit : bool
        Flag to indicate if partial results should be returned if an error occurs.
    save_all : bool
        Flag to control whether intermediate iterations should be retained.

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
        **kwargs
    ):
        SGD_Prec.__init__(self, preconditioner, **kwargs)

        self._preconditioned_lbd = False
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

    def results_warper(self, model, data, results, chrono):
        """Warp results into Sdg4vsResults object and calculate marginal likelihood, bic and ebic.

        Parameters
        ----------
        model : type[AbstractModel]
            The model used for the fitting.
        data : dict
           a dict where all additional log_likelihood arguments can be found
        results : list
            The results obtained from the fitting.
        chrono : timedelta
            The time taken for the fitting.

        Returns
        -------
        Sdg4vsResults
            An instance of Sdg4vsResults containing the results, including marginal likelihood, bic and ebic.
        """
        out = SGD_Prec.results_warper(self, model, data, results, chrono)
        chrono_start = datetime.now()
        out.update_bic(model)
        out.chrono += datetime.now() - chrono_start
        return out

    def _initialize_algo(
        self,
        model: type[AbstractModel] = None,
        theta_reals1d: jnp.ndarray = None,
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
        SGD_Prec._initialize_algo(self, model, theta_reals1d, log_likelihood_kwargs)

    # ============================================================== #

    def _one_proximal_operator(
        self, theta_reals1d, step, preconditioner
    ) -> jnp.ndarray:
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

        regularization = (
            self._lbd * jnp.diag(preconditioner)
            if self._preconditioned_lbd
            else self._lbd
        )

        return proximal_operator(
            theta_reals1d,
            stepsize=self._step_size(step),
            lbd=regularization,
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
        (theta_reals1d, grad_precond, preconditioner) = self._one_gradient_descent(
            model, log_likelihood_kwargs, theta_reals1d, step
        )

        # Proximal operator
        theta_reals1d = self._one_proximal_operator(
            theta_reals1d=theta_reals1d,
            step=step,
            preconditioner=preconditioner,
        )

        preconditioner = preconditioner if self._save_preconditionner else None
        return (theta_reals1d, grad_precond, preconditioner)
