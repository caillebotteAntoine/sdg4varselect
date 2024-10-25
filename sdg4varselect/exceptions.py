"""
Define custom exceptions.

Create by antoine.caillebotte@inrae.fr
"""


class Sdg4vsException(Exception):
    """Base of all Sdg4Varselect exceptions."""


class Sdg4vsNanError(Sdg4vsException):
    """Sdg4Varselect Error related to Nan in array."""


class Sdg4vsWrongParametrization(Sdg4vsException):
    """Sdg4Varselect Error related to wrong parametrization in models."""
