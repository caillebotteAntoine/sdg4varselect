"""tests for the Chain class"""
# pylint: disable=C0116
import pytest
import numpy as np
import sdg4varselect.miscellaneous as m


def test_chain_initialization():
    chain = m.Chain(5.0, 3, "TestChain")
    assert chain.name == "TestChain"
    assert np.array_equal(chain.data, np.array([5.0, 5.0, 5.0]))
    assert len(chain) == 3
    assert len(chain.chain) == 1

    with pytest.warns(UserWarning):
        x = m.Chain([0.5, 0.6])

    assert x.data[0] == 0.5

    with pytest.raises(TypeError) as excinfo:
        m.Chain("test")

    assert str(excinfo.value) == "x0 must be an integer or a float"

    with pytest.raises(TypeError) as excinfo:
        m.Chain(0.2, "test")

    assert str(excinfo.value) == "size must be an integer or a float"


def test_chain_reset():
    chain = m.Chain(2.0, 4, "ResetChain")
    chain.update_chain()
    chain.reset()
    assert np.array_equal(chain.data, np.array([2.0, 2.0, 2.0, 2.0]))
    assert len(chain.chain) == 1  # One initial state and one after reset


def test_chain_repr():
    chain = m.Chain(1.0, 2, "RepresentationChain")
    assert repr(chain) == "[RepresentationChain]([1. 1.])"


def test_chain_update_chain():
    chain = m.Chain(3.0, 2, "UpdateChain")
    chain.update_chain()
    assert len(chain.chain) == 2  # One initial state and one after update


def test_chain_print():
    chain = m.Chain(0.0, 2, "PrintChain")
    assert chain.print() == "[PrintChain]([0. 0.])\n previous values = []"


def test_loadbar_basic():
    result = m.loadbar("Loading: ", 0.6, 20, "*")
    assert result == "Loading: [============*        ]"


def test_loadbar_maxbar_exceeded():
    result = m.loadbar("Progress: ", 0.8, 10, "#")
    assert result == "Progress: [========#  ]"


# Example tests for loadnumber function


def test_loadnumber_basic():
    result = m.loadnumber("Progress: ", 25, 100, "%")
    assert result == "Progress:  25/100%"


def test_loadnumber_padding():
    result = m.loadnumber("Count: ", 5, 100, "%")
    assert result == "Count:   5/100%"


def test_loadbar_nbar_greater_than_maxbar():
    result = m.loadbar("Overflow: ", 1.2, 15, "@")
    assert result == "Overflow: [===============@]"


def test_step_message_basic():
    result = m.step_message(3, 10)
    assert result == " 3/10 [===============>                                   ]"


def test_step_message_large_iteration():
    result = m.step_message(15, 20)
    assert result == "15/20 [=====================================>             ]"
