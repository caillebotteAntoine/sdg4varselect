r"""


    AbstractLatentVariablesModel     AbstractModel      AbstractHDModel
                    |                /     |     \              |
                    |               /      |      \             |
                    |              /       |       \            |
                    |             /        |        \           |
        AbstractMixedEffectsModel          |       *AbstractCoxModel ------- *WeibullCoxModel
                    |                      |                 |
                    |                      |       *AbstractCoxMemJointModel
                    |                      |                 |
        *LogisticMixedEffectsModel         |       *WeibullCoxMemJointModel

    TODO : * = to be continued
"""

from .abstract.abstract_model import AbstractModel

from .abstract.abstract_latent_variables_model import AbstractLatentVariablesModel
from .abstract.abstract_mixed_effects_model import (
    AbstractMixedEffectsModel as AbstractMEM,
)

from .abstract.abstract_high_dim_model import AbstractHDModel
