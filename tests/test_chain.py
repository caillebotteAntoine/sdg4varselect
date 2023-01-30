from sdg4varselect.chain import chain
import numpy as np
import pytest


def test_chain_init():
    # name uniqueness
    chain(1, 1, name="foo")
    with pytest.raises(Exception) as except_info:
        chain(1, 1, name="foo")

    assert except_info.type is ValueError

    # numeric value in argument
    with pytest.raises(Exception) as except_info:
        chain("1", 1)

    assert except_info.type is TypeError

    # numeric size in argument
    with pytest.raises(Exception) as except_info:
        chain(1, "1")

    assert except_info.type is TypeError


def test_chain():
    x = chain(3, 2)

    assert x.data().all() == np.array([3.0, 3.0]).all()

    x.data()[0] = 2
    assert x.data()[0] == 2

    x.update_chain()
    assert x.chain()[0].all() == np.array([2, 3]).all()


def test_chain_repr():

    x = chain(3, 2)

    assert str(x) == "chain([3. 3.])"


def test_chain_print():

    x = chain(3, 2)
    assert x.print() == "chain([3. 3.])\n previous values = [[3. 3.]]"


def test_chain_size():
    x = chain(3, 2)

    assert len(x) == 2


def test_name():
    x = chain(3, 2)
    assert x.name() == "NA"

    y = chain(3, 2, name="name test")
    assert y.name() == "name test"


def test_type():
    x = chain(3, 2)
    assert x.type() == "chain"

    y = chain(3, 2, type="foo")
    assert y.type() == "foo"
