


API Documentation
***************************

This page contains the API documentation for the main classes and functions of the `Sdg4varselect` package.


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
   AbstractCoxModel
   AbstractCoxMemJointModel

   LogisticMixedEffectsModel
   PKMixedEffectsModel

   WeibullCoxModel
   CstCoxModel
   GompertzCoxModel

   WeibullHazardJM
   CstHazardJM


:mod:`\\.algo`: **Algorithms**
=======================================
Available algorithms in `Sdg4varselect`.

.. currentmodule:: sdg4varselect.algo

.. autosummary::
   :toctree: generated/algo
   :template: class.rst

   AbstractAlgoFit
   AbstractAlgoMCMC


   GradientDescentPrecond
   StochasticGradientDescentPrecond
   StochasticProximalGradientDescentPrecond

   preconditioner.AbstractPreconditioner
   preconditioner.Identity
   preconditioner.Fisher
   preconditioner.AdaGrad
   preconditioner.RMSP
   preconditioner.ADAM

:mod:`\\._mcmc`: **MCMC sampling**
=======================================
MCMC Sampler in `_mcmc`.

.. currentmodule:: sdg4varselect

.. autosummary::
   :toctree: generated/_mcmc
   :template: class.rst

   MCMC

.. autosummary::
   :toctree: generated/_chain
   :template: class.rst

   Chain

:mod:`\\.outputs`: **Results Objects**
=======================================
Available results objects in `outputs`.

.. currentmodule:: sdg4varselect

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
   Sdg4vsInfError
   Sdg4vsNanError
   Sdg4vsWrongParametrization
