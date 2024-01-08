from jax import jit
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

import itertools
from collections import namedtuple

from sdg4varselect.data_handler import Data_handler
from sdg4varselect.learning_rate import create_multi_step_size
from sdg4varselect.miscellaneous import step_message
from copy import deepcopy


@jit
def gradient_descent_fisher_preconditionner_with_mask(
    jac: jnp.ndarray,
    jac_current: jnp.ndarray,
    # fisher_identity_mixture: bool,
    step_size_approx_sto: float,
    step_size_fisher: float,
    fisher_mask: jnp.ndarray,
):
    """Compute one step of a gradient with perconditionner

    J_S = [J, O]
                  | J.T@J  0 |
    J_S.T @ J_S = | 0      0 |
                                          | J.T@J/N  0   |
    FIM = J_S.T @ J_S/N + diag(not mask) =| 0        I_p |

    """
    # Jacobian approximate
    jac = (1 - step_size_approx_sto) * jac + step_size_approx_sto * jac_current
    jac_shrink = jnp.where(fisher_mask, jac, 0)

    # Gradient
    grad = jac.mean(axis=0)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # for i in range(0, 5):
    #     grad = grad.at[i].set(0)

    # for i in range(0, 6):
    #     grad = grad.at[i].set(0)

    # for i in range(10, 14):
    #     grad = grad.at[i].set(0)

    # grad = grad.at[6].set(0)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    grad_shrink = jnp.where(fisher_mask, grad, 0)

    # ' jnp.where(jnp.array([True, False]), jnp.array([[1,2],[3,4]]),0)
    # '  = Array([[1, 0], [3, 0]])
    # '
    fim = jac_shrink.T @ jac_shrink / jac_shrink.shape[0] + jnp.diag(
        jnp.where(fisher_mask, 0, 1)
    )
    fim = step_size_fisher * fim + (1 - step_size_fisher) * jnp.eye(fim.shape[0])
    pred_grad_shrink = jnp.linalg.solve(fim, grad_shrink)

    grad_precond = jnp.where(fisher_mask, pred_grad_shrink, grad)

    return jac_shrink, fim, grad, grad_precond


@jit
def prox(
    theta: jnp.ndarray, stepsize: float, lbd: float, alpha: float = 1  # shape = (p,)
) -> jnp.ndarray:
    """apply the proximal operator on params for the elastic net panlty
    prox{stepsize, pen(lambda, alpha)} = argmin_theta (pen(theta') + 1/(2stepsize) ||theta-theta'||^2)

    gamma step size
    alpha choice between ridge and lasso
    lambda tuning parameter of elastic net
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
    HD_mask=None,
):
    return jnp.where(
        HD_mask,
        prox(theta_reals1d, stepsize, lbd, alpha),
        theta_reals1d,
    )


class NanError(Exception):
    pass


class SPG_FIM:
    def __init__(self, PRNGKey, dh: Data_handler, settings, lbd=None, alpha=1.0):
        self._data_handler = deepcopy(dh)

        self.PRNGKey = PRNGKey

        self._lbd = lbd
        self._alpha = alpha

        step_size_settings = [
            settings.step_size_grad,
            settings.step_size_approx_sto,
            settings.step_size_fisher,
        ]
        (
            self._step_size_grad,
            self._step_size_approx_sto,
            self._step_size_fisher,
        ) = create_multi_step_size(step_size_settings, num_step_size=3)

    @property
    def data(self):
        return self._data_handler.data

    @property
    def latent_variables(self):
        return self._data_handler.latent_variables

    settings = namedtuple(
        "settings",
        ("step_size_grad", "step_size_approx_sto", "step_size_fisher"),
    )
    estim_res = namedtuple("estim_res", ("theta", "FIM", "grad", "likelihood"))
    variable_selection_res = namedtuple(
        "variable_selection_res", ("estim_res", "theta", "regularization_path", "bic")
    )

    @classmethod
    def labelswitch(self, res_estim):
        res = [
            [res_estim[i][j] for i in range(len(res_estim))]
            for j in range(len(res_estim[0]))
        ]

        res = self.estim_res(
            theta=jnp.array(res[0]),
            FIM=res[2],
            grad=jnp.array(res[4]),
            likelihood=jnp.nan,
        )
        return res

    def add_mcmc(self, *args, **kwargs) -> None:
        """create a new mcmc chain and add it to the latent variable"""
        self._data_handler.add_mcmc(*args, **kwargs)

    # ============================================================== #
    def likelihood_marginal(self, model, PRNGKey, theta, size=1000):
        var_lat_sample = {}
        for var in self.latent_variables:
            var_lat_sample[var] = self.latent_variables[var].sample(
                PRNGKey,
                theta,
                size=size - 1,
                **self.data,
            )

        likelihood_kwargs = deepcopy(self.data)
        out = model.likelihood(theta, **likelihood_kwargs)
        for k in range(size - 1):
            for var in self.latent_variables:
                likelihood_kwargs[var] = var_lat_sample[var][k]

            out += model.likelihood(theta, **likelihood_kwargs)

        return out / size

    # ============================================================== #

    # @functools.partial(jit, static_argnums=0)
    def simulation(self, theta_reals1d):
        # Simulation
        for var_lat in self._data_handler.latent_variables.values():
            self.PRNGKey = var_lat.gibbs_sampler_step(
                self.PRNGKey, theta_reals1d, **self.data
            )

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
        ) = gradient_descent_fisher_preconditionner_with_mask(
            jac,
            jac_current,
            fisher_mask=FIM_MASK,
            step_size_approx_sto=step_size[1],
            step_size_fisher=step_size[2],
        )

        theta_reals1d += step_size[0] * grad_precond

        if self._lbd is not None:
            theta_reals1d = proximal_operator(
                theta_reals1d,
                step_size[0],
                lbd=self._lbd,
                alpha=self._alpha,
                HD_mask=HD_MASK,
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
        niter,
        theta_reals1d: jnp.ndarray,
        jac0,
        FIM_MASK,
        HD_MASK,
    ):
        jac = jac0
        iter = 0

        for _ in itertools.count():
            iter += 1
            print(step_message(iter, niter), end="\r")

            step_size = [
                self._step_size_grad(iter),
                self._step_size_approx_sto(iter),
                self._step_size_fisher(iter),
            ]

            (
                theta_reals1d,
                jac,
                fisher_info,
                grad,
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
                yield NanError("nan detected in theta or jac")
                break

            yield (
                theta_reals1d,
                jac,
                fisher_info,
                grad,
                grad_precond,
                # likelihood(theta_reals1d, **dh.data),
            )

    def fit(
        self,
        jac_likelihood,
        niter,
        DIM_HD,
        theta0_reals1d: jnp.ndarray,
        ntry=1,
        partial_fit=False,
    ):
        (DIM_THETA,) = theta0_reals1d.shape

        # mask for proximal operator
        HD_MASK = jnp.arange(DIM_THETA) >= DIM_THETA - DIM_HD
        # mask for fisher preconditionning
        FIM_MASK = np.invert(HD_MASK)

        # c'est pas un iterator

        jac_shape = jac_likelihood(theta0_reals1d, **self.data).shape
        out = list(
            itertools.islice(
                self.algorithm(
                    jac_likelihood,
                    niter,
                    theta0_reals1d,
                    jac0=jnp.zeros(shape=jac_shape),
                    FIM_MASK=FIM_MASK,
                    HD_MASK=HD_MASK,
                ),
                niter,
            )
        )
        flag = out[-1]
        if isinstance(flag, NanError):
            if ntry > 1:
                return self.fit(
                    jac_likelihood,
                    niter,
                    DIM_HD,
                    theta0_reals1d,
                    ntry=ntry - 1,
                    partial_fit=partial_fit,
                )
            # ie all attempts have failed
            if partial_fit:
                out.pop()  # remove error
                return out
            else:
                raise flag
        return out


def BIC(theta_HD, log_likelihood, n):
    """
    BIC = k*ln(n) - 2*ln(L)

    where :
        - k is the number of parameter estimated (ie non zero parameter in HD parameter)
        - n is the sample size
        - L the maximzed value of the likelihood function
    """
    k = (theta_HD != 0).sum(axis=1)
    assert k.shape == log_likelihood.shape

    return -2 * log_likelihood + k * jnp.log(n)


def regularization_path(one_estim, PRNGKey, model, dh, algo_settings, lbd_set):
    DIM_LD = model.DIM_LD
    PRNGKey_list = jrd.split(PRNGKey, num=len(lbd_set))

    def iter_estim():
        for i in range(len(lbd_set)):
            res_estim = one_estim(
                PRNGKey_list[i], model, dh, algo_settings, lbd=lbd_set[i]
            )
            if res_estim == NanError:
                raise NanError

            if (res_estim.theta[DIM_LD:] != 0).sum() == 0:
                for k in range(len(lbd_set) - i):
                    yield res_estim
                break
            else:
                yield res_estim

    return [res for res in iter_estim()]
