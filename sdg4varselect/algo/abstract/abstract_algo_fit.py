"""
Module for Abstract class containing common method for algorithm based on Monte-Carlo Markov Chains.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116
import itertools


from datetime import datetime
from abc import abstractmethod

import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.exceptions import sdg4vsNanError


class AbstractAlgoFit:
    """likelihood parameter estimation algorithm"""

    def __init__(self, max_iter: int):
        self._max_iter = max_iter

    @abstractmethod
    def get_likelihood_kwargs(self, data):
        """return all the needed data for the likelihood computation"""

    # ============================================================== #

    @abstractmethod
    def _initialize_algo(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """

    @abstractmethod
    def algorithm(
        self,
        model: type[AbstractModel],
        likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
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
        ntry=1,
        partial_fit=False,
        save_all=True,
    ):

        self._initialize_algo(model, self.get_likelihood_kwargs(data), theta0_reals1d)

        chrono_start = datetime.now()

        iter_algo = itertools.islice(
            self.algorithm(model, self.get_likelihood_kwargs(data), theta0_reals1d),
            self._max_iter,
        )
        if save_all:
            out = list(iter_algo)
        else:
            out = [next(iter_algo), None]
            for last in iter_algo:
                out[1] = last

        chrono_time = datetime.now() - chrono_start

        flag = out[-1]
        if isinstance(flag, sdg4vsNanError):
            if ntry > 1:
                print(f"try again because of : {flag}")
                return self.fit(
                    model,
                    data,
                    theta0_reals1d,
                    ntry=ntry - 1,
                    partial_fit=partial_fit,
                )
            # ie all attempts have failed
            if partial_fit:
                print(f"{flag} : partial result returned !")
                while isinstance(out[-1], sdg4vsNanError):
                    out.pop()  # remove error
                return self.results_warper(model, data, out, chrono_time)
            else:
                raise flag
        # every things is good
        return self.results_warper(model, data, out, chrono_time)
