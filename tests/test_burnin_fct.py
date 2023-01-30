from sdg4varselect.burnin_fct import burnin_fct
import pytest


def test_burnint_fct_init():
    # step are int
    with pytest.raises(Exception) as except_info:
        burnin_fct(step_burnin=1.2)

    assert except_info.type is TypeError

    with pytest.raises(Exception) as except_info:
        burnin_fct(step_heat=1.2)

    assert except_info.type is TypeError

    # coef are numeric
    with pytest.raises(Exception) as except_info:
        burnin_fct(coef_burnin="foo")

    assert except_info.type is TypeError
    with pytest.raises(Exception) as except_info:
        burnin_fct(coef_heat="foo")

    assert except_info.type is TypeError

    # scale is numeric
    with pytest.raises(Exception) as except_info:
        burnin_fct(scale="foo")

    assert except_info.type is TypeError


def test_burnint_fct_call():
    f = burnin_fct(5, -4, 6, 0.75, 0.5)
    value = [f(i) for i in range(10)]

    assert value == [
        0.00915781944436709,
        0.020381101989183106,
        0.045358976644706256,
        0.10094825899732769,
        0.22466448205861084,
        0.5,
        0.5,
        0.5,
        0.29730177875068026,
        0.2193456688254154,
    ]
