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
from sdg4varselect.exceptions import (
    Sdg4vsException,
    Sdg4vsInfError,
    Sdg4vsNanError,
)
from sdg4varselect.outputs import Sdg4vsResults


BASE_PARAMETERS_ALGO = """
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

    """


class AbstractAlgoFit(ABC):
    r"""Base class for implementing algorithms with common methods such as parameter initialization,
    iteration management, and results handling.

    Parameters
    ----------
    {BASE_PARAMETERS_ALGO}
    """

    def __init__(
        self,
        max_iter: int = 5000,
        partial_fit: bool = False,
        save_all: bool = True,
    ):
        self._estimate_average_length = 100
        self._skip_initialization = False
        self._starting_step = 0

        self._max_iter = max_iter - self._estimate_average_length
        self._partial_fit = partial_fit
        self._save_all = save_all

    def get_log_likelihood_kwargs(self, data: dict) -> dict:
        """Return all the needed data for the log likelihood computation

        Parameters
        ----------
        data : any
            The data required for log likelihood computation.

        Returns
        -------
        dict
            A dictionary containing the necessary data for log likelihood computation.
        """
        return data

    # ============================================================== #
    @property
    def skip_initialization(self) -> bool:
        """returns the Flag to control whether the initialization step should be skipped

        Returns
        -------
            boolean Flag to control whether the initialization step should be skipped
        """
        return self._skip_initialization

    @skip_initialization.setter
    def skip_initialization(self, skip_initialization: bool):
        self._skip_initialization = skip_initialization

    @property
    def estimate_average_length(self) -> int:
        """returns the length of the average estimation

        Returns
        -------
            an int equal to the length of the average estimation"""
        return self._estimate_average_length

    @estimate_average_length.setter
    def estimate_average_length(self, estimate_average_length: int):
        self._max_iter = self.max_iter
        self._estimate_average_length = estimate_average_length
        self.max_iter = self._max_iter

    @property
    def max_iter(self) -> int:
        """returns the maximum iteration allowed for this algorithm

        Returns
        -------
            an int equal to the maximum iteration allowed"""
        return self._max_iter + self._estimate_average_length

    @max_iter.setter
    def max_iter(self, max_iter: int):
        self._max_iter = max_iter - self._estimate_average_length

    @property
    def save_all(self) -> bool:
        """returns the Flag to control whether intermediate iterations should be retained

        Returns
        -------
            boolean Flag to control whether intermediate iterations should be retained
        """
        return self._save_all

    @save_all.setter
    def save_all(self, save_all: bool):
        self._save_all = save_all

    # ============================================================== #
    @abstractmethod
    def results_warper(
        self,
        model: type[AbstractModel],
        theta0_reals1d: jnp.ndarray,
        data: dict,
        results: list,
    ) -> Sdg4vsResults:
        """Warp results into Sdg4vsResults object.

        Parameters
        ----------
        model : type[AbstractModel]
            The model used for the fitting.
        theta0_reals1d : jnp.ndarray
            Initial parameters for the model.
        data : dict
           a dict where all additional log_likelihood arguments can be found
        results : list
            The results obtained from the fitting.

        Returns
        -------
        Sdg4vsResults
            An instance of Sdg4vsResults containing the results.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_algo(
        self,
        model: type[AbstractModel],
        theta_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray,
        log_likelihood_kwargs: dict,
    ) -> None:
        """Initialize the algorithm parameters.

        Parameters
        ----------
        model : type[AbstractModel]
            The model used for fitting.
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        freezed_components : jnp.ndarray
            boolean array indicating which parameter should not be updated
        log_likelihood_kwargs : dict
            The arguments for computing the log likelihood.
        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    # ============================================= #
    # === method to be defined in child classes === #
    # ====== to define any type of algorithm ====== #
    # ============================================= #
    @abstractmethod
    def breaking_rules(self, step, one_step_results) -> bool:
        """Abstract method to determine whether to stop the optimization process.

        This method should be implemented in subclasses to define custom stopping criteria
        for the optimization algorithm.

        Parameters
        ----------
        step : int
            The current iteration step.
        one_step_results : tuple
            The tuple returned by the _algorithm_one_step function

        Returns
        -------
        bool
            True if the stopping conditions are met, otherwise False.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _algorithm_one_step(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs,
        theta_reals1d: jnp.ndarray,
        step: int,
    ):
        """Perform one step of the algorithm.

        Parameters
        ----------
        model : type[AbstractModel]
            the model to be fitted
        log_likelihood_kwargs : dict
            a dict where all additional log_likelihood arguments can be found
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        step : int
            The current iteration step.

        Returns
        -------
        tuple
            A tuple containing the updated parameters and any additional arguments.
        """
        raise NotImplementedError

    # ============================================================== #
    # ============================================================== #
    # ============================================================== #
    def algorithm(
        self,
        model: type[AbstractModel],
        log_likelihood_kwargs: dict,
        theta_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray = None,
    ):
        """Run the iterative algorithm.

        Parameters
        ----------
        model : type[AbstractModel]
            the model to be fitted
        log_likelihood_kwargs : dict
            a dict where all additional log_likelihood arguments can be found
        theta_reals1d : jnp.ndarray
            Initial parameters for the model.
        freezed_components : jnp.ndarray, optional
            boolean array indicating which parameter components should not be updated (default is None).

        Yields
        ------
        tuple
            A tuple containing the current results.
        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        for step in itertools.count(start=self._starting_step):
            chrono = datetime.now()
            try:
                out = self._algorithm_one_step(
                    model, log_likelihood_kwargs, theta_reals1d, step
                )
            except Sdg4vsException as exc:
                yield exc
                break

            theta_reals1d = jnp.where(freezed_components, theta_reals1d, out[0])
            out = (theta_reals1d,) + out[1:]

            one_step_results = out + (datetime.now() - chrono,)
            yield one_step_results

            if self.breaking_rules(step, one_step_results):
                print(f"Stopping at step {step} because of breaking rules")
                break

    def _check_theta(self, theta):
        """
        Checks the given theta for NaN or infinite values.

        Parameters:
        -----------
        theta : jax.numpy.ndarray
            The array of theta values to be checked.

        Raises:
        -------
        Sdg4vsNanError
            If any value in theta is NaN.
        Sdg4vsInfError
            If any value in theta is infinite.
        """
        if jnp.isnan(theta).any():
            raise Sdg4vsNanError("nan detected in theta0 !")

        if jnp.isinf(theta).any():
            raise Sdg4vsInfError("inf detected in theta0 !")

    def _compute_average_estimation(self, out, out_average):
        """Compute the average estimation from the results.
        Parameters
        ----------
        out : list
            The results obtained from the fitting.
        out_average : list
            The results obtained from the average estimation.
        Returns
        -------
        list
            The updated results with the average estimation.
        """
        while len(out_average) != 0 and isinstance(out_average[-1], Sdg4vsException):
            out_average.pop()  # remove error
        if len(out_average) > 0:
            estim = jnp.array([x[0] for x in out_average]).mean(axis=0)
            out[-1] = (estim,) + out[-1][1:]
        return out

    def _iter_in_algorithm(
        self, model, log_likelihood_kwargs, theta0_reals1d, freezed_components
    ):
        """return the iteration of the algorithm
        Parameters
        ----------
        model : type[AbstractModel]
            the model to be fitted
        log_likelihood_kwargs : dict
            a dict where all additional log_likelihood arguments can be found
        theta0_reals1d : jnp.ndarray
            Initial parameters for the model.
        freezed_components : jnp.ndarray
            boolean array indicating which parameter components should not be updated (default is None).
        Returns
        -------
        iter_algo : iterator
            An iterator for the algorithm.
        """
        iter_algo = itertools.islice(
            self.algorithm(
                model, log_likelihood_kwargs, theta0_reals1d, freezed_components
            ),
            self._max_iter + self._estimate_average_length,
        )  #  creating iterators for efficient looping

        try:
            out_average = []
            if self._save_all:
                out = []
                for _ in range(self._max_iter):
                    out.append(next(iter_algo))
                    self._starting_step += 1
            else:
                out = [next(iter_algo), None]
                for _ in range(self._max_iter):
                    out[1] = next(iter_algo)
                    self._starting_step += 1

            out_average = list(iter_algo)  # remaining iterations
            if self._save_all:
                out += out_average

            self._starting_step += len(out_average)
        except StopIteration:
            pass

        return out, out_average

    def fit(
        self,
        model: type[AbstractModel],
        data: dict,
        theta0_reals1d: jnp.ndarray,
        freezed_components: jnp.ndarray = None,
    ) -> Sdg4vsResults:
        """Main method, run the algorithm according to model.

        Parameters
        ----------
            model: the model to be fitted
            data: a dict where all additional log_likelihood arguments can be found
            theta0_reals1d: starting value of parameters
            freezed_components:boolean array indicating which parameter should not be updated

        Returns
        -------
        list : Type[Sdg4vsResults]
            Depends on algorithm class
        Raises
        ------
        Sdg4vsNanError
            If NaN values are detected in starting value `theta0_reals1d` or in likelihood_kwargs.
        Sdg4vsInfError
            If Inf values are detected in starting value `theta0_reals1d` or in likelihood_kwargs.
        Sdg4vsException
            If NaN or Inf values are detected in likelihood_kwargs.
        """
        chrono_start = datetime.now()
        self._check_theta(theta0_reals1d)

        if freezed_components is None:
            freezed_components = jnp.zeros(shape=theta0_reals1d.shape, dtype=jnp.bool)

        log_likelihood_kwargs = self.get_log_likelihood_kwargs(data)
        # if _contains_nan_or_inf(log_likelihood_kwargs):
        #     raise Sdg4vsException("nan or inf detected in log_likelihood_kwargs !")

        if not self._skip_initialization:
            self._initialize_algo(
                model, theta0_reals1d, freezed_components, log_likelihood_kwargs
            )
            self._starting_step = 0

        _ntry = 1

        out, out_average = self._iter_in_algorithm(
            model, log_likelihood_kwargs, theta0_reals1d, freezed_components
        )

        flag = out[-1]
        if isinstance(flag, Sdg4vsException):
            _ntry -= 1
            if _ntry > 1:
                print(f"try again because of : {flag}")
                return self.fit(model, data, theta0_reals1d, freezed_components)
            # ie all attempts have failed
            if self._partial_fit:
                print(f"{flag} : partial result returned !")
                while len(out) != 0 and isinstance(out[-1], Sdg4vsException):
                    out.pop()  # remove error
            else:
                raise flag

        # every things is good
        if len(out) == 0:
            print("the result is empty, no iteration has been performed!")
            return out

        out = self._compute_average_estimation(out, out_average)

        out = self.results_warper(model, theta0_reals1d, data, out)
        out.chrono = datetime.now() - chrono_start
        return out
