r"""
Class hierarchy
===============

.. code-block:: text

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

"""

from .abstract.abstract_model import AbstractModel
from .abstract.abstract_latent_variables_model import AbstractLatentVariablesModel
from .abstract.abstract_mixed_effects_model import AbstractMixedEffectsModel
from .abstract.abstract_high_dim_model import AbstractHDModel
from .abstract.abstract_cox_model import AbstractCoxModel
from .abstract.abstract_cox_mem_joint_model import AbstractCoxMemJointModel
