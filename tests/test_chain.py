from sdg4varselect.chain import chain
import pytest
import numpy as np

# pytest --cov-report term-missing --cov=sdg4varselect tests/


def test_chain():
    x = chain(0.5)

    assert x.data[0] == 0.5
    assert len(x) == 1
    assert x.name == "NA"
    assert x.type == "chain"

    x = chain(0.5, 2, name="test", type="new")

    assert x.data[0] == 0.5
    assert len(x) == 2
    assert x.name == "test"
    assert x.type == "new"

    with pytest.warns(UserWarning):
        x = chain([0.5, 0.6])

    assert x.data[0] == 0.5

    with pytest.raises(TypeError) as excinfo:
        chain("test")

    assert str(excinfo.value) == "x0 must be an integer or a float"

    with pytest.raises(TypeError) as excinfo:
        chain(0.2, "test")

    assert str(excinfo.value) == "size must be an integer or a float"


def test_repr():
    x = chain(0.5, 2, name="test", type="new")
    assert str(x) == "new[test]([0.5 0.5])"


def test_print():
    x = chain(0.5, 2)
    msg = x.print()

    assert msg != 0


def test_init():
    x = chain(0.5, 2)
    x.init(1.5)
    assert (x.data == [1.5, 1.5]).all()


def test_reset_update_chain():
    x = chain(0.5, 2)

    x.data[0] = 1.5
    x.update_chain()
    assert (np.array(x.chain) == np.array([[0.5, 0.5], [1.5, 0.5]])).all()

    x.reset()
    assert (np.array(x.chain) == [np.array([0.5, 0.5])]).all()
    assert (x.data == [0.5, 0.5]).all()
