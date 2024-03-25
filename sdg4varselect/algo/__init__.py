"""
Create by antoine.caillebotte@inrae.fr

            AbstractAlgoMCMC       AbstractAlgoFit
                   |                    | 
                   |                    | 
                   |                    | 
                   |                GradientDescentFIM
                   |                /
                   |               /
                   |              /          
        StochasticGradientDescentFIM
                   |
                   |
                   |            
        StochasticProximalGradientDescentFIM

"""

from .sto_prox_grad_descent_fim import (
    StochasticProximalGradientDescentFIM as SPGD_FIM,
)
from sdg4varselect.algo.sto_grad_descent_fim import (
    StochasticGradientDescentFIM as SGD_FIM,
)

from .gradient_descent_fim import (
    GradientDescentFIMSettings as GradFimSettings,
    get_GDFIM_settings,
)
