# pylint: disable=all

import pytest
from sdg4varselect.models.abstract.abstract_model import _check_initialization
from sdg4varselect.exceptions import Sdg4vsException


class DummyModel:
    def __init__(self, initialized):
        self._is_initialized = initialized

    @property
    def is_initialized(self):
        return self._is_initialized

    @_check_initialization
    def foo(self, x):
        return x + 1


def test_check_initialization_raises_exception_when_not_initialized():
    model = DummyModel(initialized=False)
    with pytest.raises(Sdg4vsException) as excinfo:
        model.foo(1)
    assert "The model has not been initiated" in str(excinfo.value)


def test_check_initialization_allows_call_when_initialized():
    model = DummyModel(initialized=True)
    result = model.foo(2)
    assert result == 3
