import sdg4varselect.miscellaneous as miscel
import pytest


def test_loadbar():
    assert miscel.loadbar("", 1.3, 5) == "[=====>]"


def test_loadnumber():
    assert miscel.loadnumber("", 2, 42) == " 2/42"
    assert miscel.loadnumber("", 24, 42) == "24/42"


def test_step_message():
    assert miscel.step_message(24, 42, maxbar=10) == "24/42 [=====>     ]"

    assert miscel.step_message(41, 42, maxbar=10) == "41/42 [=========> ]\n"


def test_time2string():
    assert miscel.time2string(2.5) == "2.500s"
    assert miscel.time2string(0.6) == "0.600s"
    assert miscel.time2string(2.3e-4) == "0.230ms"
    assert miscel.time2string(2.3e-3) == "2.300ms"
    assert miscel.time2string(2.2e-7) == "0.220µs"
    assert miscel.time2string(2.2e-6) == "2.200µs"
    assert miscel.time2string(2.1e-10) == "0.210ns"
