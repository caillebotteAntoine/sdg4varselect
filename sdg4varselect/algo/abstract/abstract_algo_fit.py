# pylint: disable=C0116
import itertools


from datetime import datetime
from abc import abstractmethod

import jax.numpy as jnp


from sdg4varselect.models.abstract.abstract_model import AbstractModel
from sdg4varselect.exceptions import sdg4vsNanError


class AbstractAlgoFit:
    """vraisseblance parameter estimation algorithm"""

    def __init__(self, max_iter: int):
        self._max_iter = max_iter

    @abstractmethod
    def get_likelihood_kwargs(self, data):
        """return all the needed data"""

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
        """iterative algorithm"""

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
    ):

        self._initialize_algo(model, self.get_likelihood_kwargs(data), theta0_reals1d)

        chrono_start = datetime.now()
        out = list(
            itertools.islice(
                self.algorithm(model, self.get_likelihood_kwargs(data), theta0_reals1d),
                self._max_iter,
            )
        )
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
        return self.results_warper(model, data, out, chrono_time)
