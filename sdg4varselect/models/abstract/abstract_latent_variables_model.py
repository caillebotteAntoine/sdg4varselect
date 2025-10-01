"""
This module defines the `AbstractLatentVariablesModel` class, an abstract base for models with latent variables,
as well as several functions for formatting cov values, sampling latent variables, and calculating log-likelihoods.

The `AbstractLatentVariablesModel` class is designed to support models with latent variables, providing properties
and abstract methods that subclasses must implement. It defines required attributes and structure for models that
use latent variables with a Gaussian prior.

Functions in this module include:
- `sample_latent`: Samples from a multivariate normal distribution using provided covariance.
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
from jax.scipy.special import logsumexp

from sdg4varselect.models.abstract.abstract_model import AbstractModel


def _mean_formatting(mean, size):
    if isinstance(mean, (tuple, list)):
        mean = jnp.hstack(mean)

    z = jnp.zeros(shape=(size - mean.shape[0],))
    mean = jnp.concatenate([mean, z])
    return mean


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
    mean = jnp.zeros(shape=(D,))  # Default mean is zero
    cov = _cov_formatting(params.cov_latent)

    shape = (N,)  # mean.shape[0])
    return jrd.multivariate_normal(prngkey, mean=mean, cov=cov, shape=shape)


@jit
def multivariate_normal_log_pdf(
    x: jnp.ndarray, mean: jnp.ndarray, cov: jnp.ndarray
) -> jnp.ndarray:
    """Compute the log probability of a Gaussian prior with covariance matrix.

        sqrt((2pi)^D det(cov)) * exp[-1/2(x-m)^T cov^-1 (x-m)]

    Parameters
    ----------
    x : jnp.ndarray
        Observed data of shape `(N, D)`.
    mean : jnp.ndarray
        Mean of the latent variables with shape `(D,)`.
    cov : jnp.ndarray
        Covariance matrix of the latent variables with shape `(D, D)`.

    Returns
    -------
    jnp.ndarray
        Log probability values of shape `(N,)`.
    """
    N, D = x.shape
    assert mean.shape in ((N, D), (D,))
    assert cov.shape == (D, D)

    x_sub_mean = x - mean

    out = (
        jnp.linalg.slogdet(cov)[1]  # log du det
        + D * jnp.log(2 * jnp.pi)
        + ((x_sub_mean @ jnp.linalg.inv(cov)) * x_sub_mean).sum(axis=1)
    )

    assert out.shape == (N,)
    return -out / 2


class AbstractLatentVariablesModel(ABC):
    """Abstract class for models with latent variables.

    Defines essential attributes for latent variables, including their names and sizes.
    Subclasses should specify methods for computing likelihoods with and without priors.

    latent variables parameter must be named  and cov_latent.

    Latent variables must be arranged in a precise order:
        those with a zero mean, then those with a non-zero mean.
    All latent variables are defined with a variance and covariance defined in
    cov_latent as follows:

    *   cov_latent must have a Matrix parametrization type.
        cov_latent will have a shape (D,D).

    *   mean_latent can be Scalar parametrization of Shape (1,),
        Tuple or Namedtuple of Scalar parametrization.

        mean_latent shape can't exceed (D,)

    Methods
    -------
    add_latent_variables(name)
        Add a new latent variable to the model.
    get_mean_latent(params, **kwargs) -> jnp.ndarray
        Compute the mean of the latent variables.
    only_prior(params, **kwargs) -> jnp.ndarray
        Compute log-likelihood with only the Gaussian prior.
    log_likelihood_only_prior(theta_reals1d, **kwargs) -> jnp.ndarray
        Abstract method to compute log-likelihood with only the Gaussian prior.
    log_likelihood_without_prior(theta_reals1d, **kwargs) -> jnp.ndarray
        Abstract method to compute the log-likelihood without Gaussian prior.
    sample_latent_variables(params_star, prngkey, **kwargs) -> tuple[dict, dict]
        Sample latent variables for the model sampling.
    sample(params_star, prngkey, **kwargs) -> tuple[dict, dict]
        Sample latent variables for the model sampling.

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

    @functools.partial(jit, static_argnums=0)
    def get_mean_latent(
        self,
        params,
        **kwargs,  # pylint: disable=unused-argument
    ) -> jnp.ndarray:
        """
        Compute the mean of the latent variables.
        This method formats the mean_latent parameter to ensure it matches the expected size
        Parameters
        ----------
        params : object
            Contains `mean_latent` and `cov_latent`.
        **kwargs : dict
            Additional data to be passed to the mean computation.

        Returns
        -------
        jnp.ndarray
            Mean of the latent variables.
        """
        return _mean_formatting(params.mean_latent, size=params.cov_latent.shape[0])

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
            additional data containing the latent variables and needed variables for mean computation.

        Returns
        -------
        jnp.ndarray
            Log-likelihood with only the Gaussian prior.
        """
        data = [kwargs[name] for name in self._latent_variables_name]
        cov = _cov_formatting(params.cov_latent)
        mean = self.get_mean_latent(params, **kwargs)

        return multivariate_normal_log_pdf(x=jnp.array(data).T, mean=mean, cov=cov)

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
    def sample_latent_variables(self, params_star, prngkey, **kwargs):
        """Sample latent variables for the model sampling

        Parameters
        ----------
        params_star : object
            parameter used to sample the latent variables
        prngkey : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample randomly the latent variables
        **kwargs : dict
            Additional data needed for mean computation.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
                - dict: empty by default.
                - dict: Simulated latent variables.
        """
        D = len(self.latent_variables_name)
        sim_latent = _sample_latent(prngkey, params_star, N=self.latent_variables_size)
        sim_latent += self.get_mean_latent(params_star, **kwargs)

        assert sim_latent.shape == (self.latent_variables_size, D)

        sim = dict(
            zip(
                self.latent_variables_name,
                [sim_latent[:, i] for i in range(D)],
            )
        )
        return sim

    def sample(self, params_star, prngkey, **kwargs) -> tuple[dict, dict]:
        """Sample latent variables for the model sampling

        Parameters
        ----------
        params_star : object
            parameter used to sample the latent variables
        prngkey : jax.random.PRNGKey
            A PRNG key, consumable by random functions used to sample randomly the latent variables
        **kwargs : dict
            Additional data needed for mean computation.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
                - dict: empty by default.
                - dict: Simulated latent variables.
        """
        sample_key, prngkey = jrd.split(prngkey, num=2)
        sim_latent = self.sample_latent_variables(params_star, sample_key, **kwargs)

        return {}, sim_latent


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
    var_lat_sample = model.sample_latent_variables(
        params, sample_key, **data
    )  # dict with simulated latent variables

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
