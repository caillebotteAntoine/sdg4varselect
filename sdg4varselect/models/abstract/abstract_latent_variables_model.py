"""
This module defines the `AbstractLatentVariablesModel` class, an abstract base for models with latent variables,
as well as several functions for formatting mean values, sampling latent variables, and calculating log-likelihoods.

The `AbstractLatentVariablesModel` class is designed to support models with latent variables, providing properties
and abstract methods that subclasses must implement. It defines required attributes and structure for models that
use latent variables with a Gaussian prior.

Functions in this module include:
- `sample_latent`: Samples from a multivariate normal distribution using provided mean and covariance.
- `log_gaussian_prior_cov`: Computes the log probability of a Gaussian prior with a specified covariance.
- `log_likelihood_marginal`: Computes the marginal log-likelihood for model instances by sampling latent variables.

Create by antoine.caillebotte@inrae.fr
"""

import functools
from typing import Union, Type

from abc import ABC, abstractmethod

from jax import jit
import jax.numpy as jnp
import jax.random as jrd
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp

from sdg4varselect.models.abstract.abstract_model import AbstractModel


def _cov_formatting(cov):
    if isinstance(cov, (tuple, list)):
        cov = jnp.hstack(cov)

    if len(cov.shape) == 1:
        return jnp.diag(cov)

    return cov


def _sample_latent(prngkey, params, N):
    """
    Generate samples from a multivariate normal distribution using latent mean and covariance.

    Parameters
    ----------
    prngkey : jax.random.PRNGKey
        Random key for reproducibility.
    params : object
        Contains the attribute `cov_latent`.
    N : int
        Number of samples to draw.

    Returns
    -------
    jnp.ndarray
        Generated samples with shape `(N, D)`.
    """
    D = params.cov_latent.shape[0]
    mean = jnp.zeros(shape=(D,))
    cov = _cov_formatting(params.cov_latent)

    shape = (N,)  # mean.shape[0])
    return jrd.multivariate_normal(prngkey, mean=mean, cov=cov, shape=shape)


class AbstractLatentVariablesModel(ABC):
    """Abstract class for models with latent variables.

    Defines essential attributes for latent variables, including their names and sizes.
    Subclasses should specify methods for computing likelihoods with and without priors.

    latent variables parameter must be named  and cov_latent.

    Latent variables must be arranged in a precise order:
        those with a zero mean, then those with a non-zero mean.
    All latent variables are defined with a variance and covariance defined in
    cov_latent as follows:

    -   cov_latent must have a Matrix parametrization type.
        cov_latent will have a shape (D,D).

    -   mean_latent can be Scalar parametrization of Shape (1,),
        Tuple or Namedtuple of Scalar parametrization.

        mean_latent shape can't exceed (D,)

    Attributes
    ----------
    _latent_variables_name : list of str
        Names of latent variables.
    _latent_variables_size : int
        Size of latent variables.
    """

    def __init__(self, size=int):
        self._latent_variables_name = []
        self._latent_variables_size = size

    @property
    def latent_variables_name(self):
        """list of str: Names of latent variables."""
        return self._latent_variables_name

    @property
    def latent_variables_size(self):
        """int: Size of latent variables."""
        return self._latent_variables_size

    def add_latent_variables(self, name):
        """Add a new latent variable to the model.

        Parameters
        ----------
        name : str
            The name of the new latent variable

        Raises
        ------
        KeyError
            If an latent variable with the same name already exists.
        """
        if name in self._latent_variables_name:
            raise KeyError(name + " all ready exist as latent variables.")
        self._latent_variables_name += [name]

    # ============================================================== #
    @functools.partial(jit, static_argnums=0)
    def only_prior(self, params, **kwargs) -> jnp.ndarray:
        """
        Compute log-likelihood with only the Gaussian prior.

        Parameters
        ----------
        params : object
            Contains attribute `cov_latent`.
        **kwargs : dict
            additional data to be pass to the log-likelihood

        Returns
        -------
        jnp.ndarray
            Log-likelihood with only the Gaussian prior.
        """
        data = [kwargs[name] for name in self._latent_variables_name]
        cov = _cov_formatting(params.cov_latent)
        mean = jnp.zeros(shape=(cov.shape[0],))

        return multivariate_normal.logpdf(x=jnp.array(data).T, mean=mean, cov=cov)

    @abstractmethod
    def log_likelihood_only_prior(self, theta_reals1d, **kwargs) -> jnp.ndarray:
        """Compute log-likelihood with only the Gaussian prior.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Parameters used to the log-likelihood computation.
        **kwargs : dict
            a dict where all additional log_likelihood arguments can be found.

        Returns
        -------
        jnp.ndarray
            Log-likelihood values with only the Gaussian prior.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def log_likelihood_without_prior(self, theta_reals1d, **kwargs) -> jnp.ndarray:
        """Compute the log-likelihood without Gaussian prior.

        Parameters
        ----------
        theta_reals1d : jnp.ndarray
            Parameters used to the log-likelihood computation.
        **kwargs : dict
            Additional keyword arguments used in the mixed_effect_function

        Returns
        -------
        jnp.ndarray
            Log-likelihood values for each individual without Gaussian prior.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    # ============================================================== #
    def sample(self, params_star, prngkey) -> tuple[dict, dict]:
        """Sample latent variables for the model sampling

        Parameters
        ----------
        params_star : object
            parameter used to sample the latent variables
        prngkey : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample randomly the latent variables

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
                - dict: empty by default.
                - dict: Simulated latent variables.
        """
        key, prngkey = jrd.split(prngkey, num=2)

        D = len(self.latent_variables_name)

        sim_latent = _sample_latent(
            key, params_star, N=self.latent_variables_size
        )  # jnp.array shape ?= (N,D)

        assert sim_latent.shape == (self.latent_variables_size, D)

        sim = dict(
            zip(
                self.latent_variables_name,
                [sim_latent[:, i] for i in range(D)],
            )
        )

        return {}, sim


@functools.partial(jit, static_argnums=0)
def _new_log_likelihood(
    model: Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]],
    sample_key,
    data,
    theta_reals1d: jnp.ndarray,
) -> jnp.ndarray:
    """Compute likelihood for a model instance.

    Parameters
    ----------
    model : Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]]
        Model instance.
    sample_key : jax.random.PRNGKey
        A PRNG key, consumable by random functions used to sample latent variables.
    data : dict
        a dict where all additional log_likelihood arguments can be found.
    theta_reals1d : jnp.ndarray
        Parameters passed to the log-likelihood function.

    Returns
    -------
    jnp.ndarray
        Computed likelihood.
    """

    params = model.parametrization.reals1d_to_params(theta_reals1d)
    sim_latent = _sample_latent(sample_key, params=params, N=model.N)

    var_lat_sample = dict(
        zip(
            model.latent_variables_name,
            [sim_latent[:, i] for i in range(sim_latent.shape[1])],
        )
    )

    return model.log_likelihood_without_prior(
        theta_reals1d, **data, **var_lat_sample
    )  # log(f(Y|phi_sim)) ; shape = (N,)


def log_likelihood_marginal(
    model: Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]],
    prngkey,
    data: dict,
    theta_reals1d: jnp.ndarray,
    size=1000,
) -> jnp.ndarray:
    """
    Compute the marginal log-likelihood.

    Parameters
    ----------
    model : Union[Type[AbstractModel], Type[AbstractLatentVariablesModel]]
        Model instance.
    prngkey : jax.random.PRNGKey
        A PRNG key, consumable by random functions used to sample latent variables.
    data : dict
        a dict where all additional log_likelihood arguments can be found.
    theta_reals1d : jnp.ndarray
        Parameters passed to the log-likelihood function.
    size : int, optional
        Number of simulations for marginalization, by default 1000.

    Returns
    -------
    jnp.ndarray
        Marginal log-likelihood.
    """

    out = []
    for _ in range(size):
        prngkey, sample_key = jrd.split(prngkey, 2)
        out.append(
            _new_log_likelihood(model, sample_key, data, theta_reals1d)
        )  # log(f(Y|phi_sim)) ; shape = (N,)

    # f(Y|phi_sim) ; shape = (N,)
    value = logsumexp(jnp.array(out), b=1 / len(out), axis=0).sum()
    value_old = logsumexp(jnp.array(out[:-2]), b=1 / len(out[:-2]), axis=0).sum()

    n_simu = len(out)
    while n_simu < size * 2 and (abs(value / value_old - 1.0) >= 1e-2).all():
        print(abs(value - value_old))
        for _ in range(100):
            n_simu += 1
            prngkey, sample_key = jrd.split(prngkey, 2)
            out.append(
                _new_log_likelihood(model, sample_key, data, theta_reals1d)
            )  # log(f(Y|phi_sim)) ; shape = (N,)

        value_old = value
        # f(Y|phi_sim) ; shape = (N,)
        value = logsumexp(jnp.array(out), b=1 / len(out), axis=0).sum()
    return value
