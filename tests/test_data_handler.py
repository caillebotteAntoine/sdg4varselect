"""tests for the DataHandler class"""
# pylint: disable=C0116, W0621
import pytest
from sdg4varselect._data_handler import DataHandler


@pytest.fixture
def dh():
    return DataHandler()


def test_init(dh):
    assert isinstance(dh.latent_variables, dict)
    assert isinstance(dh.data, dict)


def test_deepcopy(dh):
    copied_handler = dh.deepcopy()
    assert copied_handler.latent_variables == dh.latent_variables
    assert copied_handler.data == dh.data
    assert copied_handler is not dh  # Check if it's a deep copy


def test_add_data(dh):
    dh.add_data(variable1=42, variable2="example")
    assert dh.data == {"variable1": 42, "variable2": "example"}

    with pytest.raises(KeyError):
        dh.add_data(variable1=100)  # Already exists, should raise KeyError


def test_update_data(dh):
    dh.add_data(variable1=42, variable2="example")
    dh.update_data(variable1=100)

    with pytest.raises(KeyError):
        dh.update_data(variable3="new_variable")  # do not exists, should raise KeyError

    assert dh.data == {
        "variable1": 100,
        "variable2": "example",
    }

    dh.add_mcmc(
        0,
        sd=2,
        size=10,
        likelihood=lambda x: x,
        name="chain1",
    )
    with pytest.raises(KeyError):
        dh.update_data(
            chain1="updated_variable"
        )  # Cannot update latent variables, should raise KeyError


def test_add_mcmc(dh):
    dh.add_mcmc(
        0,
        sd=2,
        size=10,
        likelihood=lambda x: x,
        name="chain1",
    )
    assert "chain1" in dh.latent_variables

    with pytest.raises(KeyError):
        dh.add_mcmc(
            0,
            sd=2,
            size=10,
            likelihood=lambda x: x,
            name="chain1",
        )  # Already exists, should raise KeyError


def test_latent_variables_property(dh):
    assert isinstance(dh.latent_variables, dict)


def test_data_property(dh):
    assert isinstance(dh.data, dict)
