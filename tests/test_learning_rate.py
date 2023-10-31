from sdg4varselect.learning_rate import learning_rate
import pytest
import numpy as np

# pytest --cov-report term-missing --cov=sdg4varselect tests/


def test_chain():
    x = learning_rate(0.5, 1, 2, 0.7, 3, 10, 1)

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
