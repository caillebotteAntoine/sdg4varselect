r"""
Class hierarchy
===============

            AbstractAlgoMCMC       AbstractAlgoFit
                   |                    |
                   |                    |
                   |                    |
                   |                GradientDescentPrecond
                   |                /
                   |               /
                   |              /
        StochasticGradientDescentPrecond
                   |
                   |
                   |
        StochasticProximalGradientDescentPrecond

"""

from .abstract.abstract_algo_fit import AbstractAlgoFit
from .abstract.abstract_algo_mcmc import AbstractAlgoMCMC

from .gradient_descent_precond import GradientDescentPrecond
from .sto_grad_descent_precond import StochasticGradientDescentPrecond
from .sto_prox_grad_descent_precond import StochasticProximalGradientDescentPrecond
from .preconditioner import preconditioner_factory
