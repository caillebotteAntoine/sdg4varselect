"""
Module for Abstract class containing common method for algorithm based on Monte-Carlo Markov Chains.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116
import itertools


from datetime import datetime
from abc import ABC, abstractmethod

import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.exceptions import sdg4vsNanError


class AbstractAlgoFit(ABC):
    """log likelihood parameter estimation algorithm"""

    def __init__(self, max_iter: int):
        self._max_iter = max_iter

    @abstractmethod
    def get_log_likelihood_kwargs(self, data):
        """return all the needed data for the log likelihood computation"""

    # ============================================================== #
    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._max_iter = max_iter

    @abstractmethod
    def _initialize_algo(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """

    @abstractmethod
    def algorithm(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray,
    ):
        """iterative algorithm, must be iterable"""

    @abstractmethod
    def results_warper(self, model, data, results, chrono):
        """warp results"""

    def fit(
        self,
        model: type[AbstractModel],
        data,
        theta0_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray = None,
        ntry=1,
        partial_fit=False,
        save_all=True,
    ):

        chrono_start = datetime.now()
        if freezed_components is None:
            freezed_components = jnp.zeros(shape=theta0_reals1d.shape, dtype=jnp.bool)

        log_likelihood_kwargs = self.get_log_likelihood_kwargs(data)

        self._initialize_algo(model, log_likelihood_kwargs, theta0_reals1d)

        iter_algo = itertools.islice(
            self.algorithm(
                model, log_likelihood_kwargs, theta0_reals1d, freezed_components
            ),
            self._max_iter,
        )

        if save_all:
            out = list(iter_algo)
        else:
            out = [next(iter_algo), None]
            for last in iter_algo:
                out[1] = last

        flag = out[-1]
        if isinstance(flag, sdg4vsNanError):
            if ntry > 1:
                print(f"try again because of : {flag}")
                return self.fit(
                    model,
                    data,
                    theta0_reals1d,
                    freezed_components,
                    ntry=ntry - 1,
                    partial_fit=partial_fit,
                    save_all=save_all,
                )
            # ie all attempts have failed
            if partial_fit:
                print(f"{flag} : partial result returned !")
                while isinstance(out[-1], sdg4vsNanError):
                    out.pop()  # remove error
            else:
                raise flag
        # every things is good

        out = self.results_warper(
            model, data, out, chrono=datetime.now() - chrono_start
        )
        return out
