# pylint: disable=C0116
import itertools
from collections import namedtuple

from typing import Optional

import jax.numpy as jnp
import numpy as np


from sdg4varselect.learning_rate import create_multi_step_size
from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.outputs import SDGResults
from sdg4varselect._data_handler import DataHandler

from sdg4varselect.algo.abstract_algo_mcmc import AbstractAlgoMCMC


from sdg4varselect.algo.stochastic_gradient_descent_utils import (
    gradient_descent_fisher_preconditionner_with_mask as gd_fim_precond_mask,
    proximal_operator,
)


class SPG_FIM(AbstractAlgoMCMC):
    SPGfimSettings = namedtuple(
        "SPGfimSettings",
        ("step_size_grad", "step_size_approx_sto", "step_size_fisher", "max_iter"),
    )

    def __init__(
        self,
        prngkey,
        dh: DataHandler,
        settings: SPGfimSettings,
        lbd: Optional[float] = None,
        alpha: Optional[float] = 1.0,
    ):
        super().__init__(prngkey, dh)

        self._lbd = lbd
        self._alpha = alpha
        self._max_iter = settings.max_iter

        step_sizes = create_multi_step_size(list(settings)[:-1], num_step_size=3)
        (
            self._step_size_grad,
            self._step_size_approx_sto,
            self._step_size_fisher,
        ) = (
            step_sizes[0],
            step_sizes[1],
            step_sizes[2],
        )

        heating_list = [
            settings.step_size_approx_sto["heating"],
            settings.step_size_fisher["heating"],
            settings.step_size_grad["heating"],
        ]

        self._heating = (
            jnp.inf
            if len(heating_list) == 0
            else max([h for h in heating_list if h is not None])
        )

    # ============================================================== #

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

        (
            jac,
            fisher_info,
            grad,
            grad_precond,
        ) = gd_fim_precond_mask(
            jac,
            jac_current,
            fisher_mask=FIM_MASK,
            step_size_approx_sto=step_size[1],
            step_size_fisher=step_size[2],
        )

        grad_precond *= step_size[0]
        theta_reals1d += grad_precond

        if self._lbd is not None:
            theta_reals1d = proximal_operator(
                theta_reals1d,
                step_size[0],
                lbd=self._lbd,
                alpha=self._alpha,
                hd_mask=HD_MASK,
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
        theta_reals1d: jnp.ndarray,
        jac0,
        FIM_MASK,
        HD_MASK,
        eps=1e-3,
    ):
        jac = jac0
        for step in itertools.count():
            # print(step_message(step, self._max_iter), end="\r")

            step_size = [
                self._step_size_grad(step),
                self._step_size_approx_sto(step),
                self._step_size_fisher(step),
            ]

            (
                theta_reals1d,
                jac,
                fisher_info,
                _,
                grad_precond,
            ) = self.SPG_FIM_one_iter(
                jac_likelihood,
                theta_reals1d,
                jac,
                FIM_MASK,
                HD_MASK,
                step_size,
            )

            if jnp.isnan(theta_reals1d).any():
                yield sdg4vsNanError("nan detected in theta or jac")
                break

            yield (theta_reals1d, fisher_info, grad_precond, jnp.nan)

            if step > self._heating and jnp.sqrt((grad_precond**2).sum()) < eps:
                break

    def fit(
        self,
        jac_likelihood,
        DIM_HD,
        theta0_reals1d: jnp.ndarray,
        ntry=1,
        partial_fit=False,
        eps=1e-3,
    ):
        (DIM_THETA,) = theta0_reals1d.shape

        # mask for fisher preconditionning
        FIM_MASK = jnp.arange(DIM_THETA) < DIM_THETA
        # mask for proximal operator
        HD_MASK = jnp.arange(DIM_THETA) >= DIM_THETA - DIM_HD

        jac_shape = jac_likelihood(theta0_reals1d, **self.data).shape
        out = list(
            itertools.islice(
                self.algorithm(
                    jac_likelihood,
                    theta0_reals1d,
                    jac0=jnp.zeros(shape=jac_shape),
                    FIM_MASK=FIM_MASK,
                    HD_MASK=HD_MASK,
                    eps=eps,
                ),
                self._max_iter,
            )
        )
        flag = out[-1]
        if isinstance(flag, sdg4vsNanError):
            if ntry > 1:
                print(f"try again because of : {flag}")
                return self.fit(
                    jac_likelihood,
                    DIM_HD,
                    theta0_reals1d,
                    ntry=ntry - 1,
                    partial_fit=partial_fit,
                )
            # ie all attempts have failed
            if partial_fit:
                print(f"{flag} : partial result returned !")
                while isinstance(out[-1], sdg4vsNanError):
                    out.pop()  # remove error
                return SDGResults.new_from_list(out)
            else:
                raise flag
        return SDGResults.new_from_list(out)
