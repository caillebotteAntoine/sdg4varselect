API Documentation
***************************

TODO

:mod:`\\.models`: **Models**
=======================================
Available models in `Sdg4varselect`.


.. currentmodule:: sdg4varselect.models

.. autosummary::
   :toctree: generated/models
   :template: class.rst

   AbstractModel
   AbstractLatentVariablesModel
   AbstractMixedEffectsModel
   AbstractHDModel

:mod:`\\.algo`: **Algorithms**
=======================================
Available algorithms in `Sdg4varselect`.

.. currentmodule:: sdg4varselect.algo

.. autosummary::
   :toctree: generated/algo
   :template: class.rst

   ./abstract/AbstractAlgoFit
   ./abstract/AbstractAlgoMCMC

   ./preconditionner/AbstractPreconditioner
   ./preconditionner/Fisher
   ./preconditionner/AdaGrad

   GradientDescentPrecond
   StochasticGradientDescentPrecond
   StochasticProximalGradientDescentPrecond

:mod:`\\._mcmc`: **MCMC sampling**
=======================================
MCMC Sampler in `_mcmc`.

.. currentmodule:: sdg4varselect

.. autosummary::
   :toctree: generated/mcmc
   :template: class.rst

   _chain.Chain
   _mcmc.MCMC

:mod:`\\.outputs`: **Results Objects**
=======================================
Available results objects in `outputs`.

.. currentmodule:: sdg4varselect.outputs

.. autosummary::
   :toctree: generated/outputs
   :template: class.rst

   Sdg4vsResults
   GDResults
   SGDResults
   MultiGDResults
   RegularizationPath

:mod:`\\.exceptions`: **Exceptions**
=======================================
Available exceptions in `Sdg4varselect`.

.. currentmodule:: sdg4varselect.exceptions

.. autosummary::
   :toctree: generated/exceptions
   :template: class.rst

   Sdg4vsException
   Sdg4vsNanError
   Sdg4vsWrongParametrization
