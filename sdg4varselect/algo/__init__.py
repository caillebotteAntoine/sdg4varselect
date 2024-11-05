"""
Create by antoine.caillebotte@inrae.fr

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
        StochasticProximalGradientDescentFIM

"""

from .gradient_descent_precond import (
    GradientDescentPrecond as GD_Prec,
)

from .sto_grad_descent_precond import (
    StochasticGradientDescentPrecond as SGD_Prec,
)

from .sto_prox_grad_descent_precond import (
    StochasticProximalGradientDescentPrecond as SPGD_Prec,
)
