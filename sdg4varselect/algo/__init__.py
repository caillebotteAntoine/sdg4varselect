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

from .gradient_descent_fim import (
    GradientDescentFIM as GD_FIM,
    # GradientDescentFIMSettings as GradFimSettings,
    # get_gdfim_settings,
)

from .sto_grad_descent_fim import (
    StochasticGradientDescentFIM as SGD_FIM,
)


from .sto_prox_grad_descent_fim import (
    StochasticProximalGradientDescentFIM as SPGD_FIM,
)
