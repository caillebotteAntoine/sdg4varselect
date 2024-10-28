"""
This module defines the `AbstractAlgoFit` class, an abstract base class for implementing algorithms
focused on log-likelihood parameter estimation for various models.

Create by antoine.caillebotte@inrae.fr
"""

import itertools

from datetime import datetime
from abc import ABC, abstractmethod

import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.exceptions import Sdg4vsNanError


class AbstractAlgoFit(ABC):
    """Base class for implementing algorithms with common methods such as parameter initialization,
    iteration management, and results handling.

    Attributes
    ----------
    _max_iter : int
        Maximum number of iterations allowed for the algorithm.

    _ntry : int
        Number of attempts to retry the algorithm if an error is encountered.

    _ntry_max : int
        Maximum number of retry attempts to reset `_ntry` after each algorithm run.

    _partial_fit : bool
        Flag to indicate if partial results should be returned if an error occurs.

    _save_all : bool
        Flag to control whether intermediate iterations should be retained.

    """

    def __init__(
        self,
        max_iter: int,
        ntry: int = 1,
        partial_fit: bool = False,
        save_all: bool = True,
    ):
        self._max_iter = max_iter
        self._ntry = ntry
        self._ntry_max = ntry
        self._partial_fit = partial_fit
        self._save_all = save_all

    @abstractmethod
    def get_log_likelihood_kwargs(self, data: dict):
        """return all the needed data for the log likelihood computation

        Parameters
        ----------
            data: input data dict for loglikelihood computation"""
        raise NotImplementedError

    # ============================================================== #
    @property
    def max_iter(self) -> int:
        """returns the maximum iteration allowed for this algorithm

        Returns
        -------
            an int equal to the maximum iteration allowed"""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        self._max_iter = max_iter

    @abstractmethod
    def _initialize_algo(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs: dict,
        theta_reals1d: jnp.ndarray,
    ) -> None:
        """
        Initialize the algorithm
        """
        raise NotImplementedError

    @abstractmethod
    def algorithm(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs: dict,
        theta_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray,
    ):
        """iterative algorithm, must be iterable

        Parameters
        ----------
            model: the model to be fitted
            log_likelihood_kwargs: a dict where all additional log_likelihood arguments can be found
            theta_reals1d: current value of parameters
            freezed_components: boolean array indicating which parameter should not be updated
        """
        raise NotImplementedError

    @abstractmethod
    def results_warper(
        self, model: type[AbstractModel], data: dict, results: list, chrono: int
    ):
        """warp results

        Parameters
        ----------
            model: the model to be fitted
            data: a dict where all additional log_likelihood arguments can be found
            results: results of the algorithm
            chrono: time elapsed while solving algorithm

        Returns
        -------
            Depends on algorithm class
        """
        raise NotImplementedError

    def fit(
        self,
        model: type[AbstractModel],
        data: dict,
        theta0_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray = None,
    ):  # pylint :disable=missing-return-type-doc
        """Main method, run the algorithm according to model.

        Parameters
        ----------
            model: the model to be fitted
            data: a dict where all additional log_likelihood arguments can be found
            theta0_reals1d: starting value of parameters
            freezed_components:boolean array indicating which parameter should not be updated

        Returns
        -------
        list
            Depends on algorithm class
        """
        chrono_start = datetime.now()
        if freezed_components is None:
            freezed_components = jnp.zeros(shape=theta0_reals1d.shape, dtype=jnp.bool)

        log_likelihood_kwargs = self.get_log_likelihood_kwargs(data)

        self._initialize_algo(model, log_likelihood_kwargs, theta0_reals1d)
        self._ntry = self._ntry_max

        iter_algo = itertools.islice(
            self.algorithm(
                model, log_likelihood_kwargs, theta0_reals1d, freezed_components
            ),
            self._max_iter,
        )

        if self._save_all:
            out = list(iter_algo)
        else:
            out = [next(iter_algo), None]
            for last in iter_algo:
                out[1] = last

        flag = out[-1]
        if isinstance(flag, Sdg4vsNanError):
            self._ntry -= 1
            if self._ntry > 1:
                print(f"try again because of : {flag}")
                return self.fit(model, data, theta0_reals1d, freezed_components)
            # ie all attempts have failed
            if self._partial_fit:
                print(f"{flag} : partial result returned !")
                while len(out) != 0 and isinstance(out[-1], Sdg4vsNanError):
                    out.pop()  # remove error
            else:
                raise flag
        # every things is good

        out = self.results_warper(
            model, data, out, chrono=datetime.now() - chrono_start
        )
        return out