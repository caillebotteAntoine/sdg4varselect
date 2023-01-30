from sdg4varselect.MCMC import MCMC_chain
import numpy as np
import pytest


def test_mcmc_init():
    # array value in argument
    with pytest.raises(Exception) as except_info:
        MCMC_chain(
            10,
            5,
            sd=1,
            mean=5,
            variance=np.array([0.5]),
        )
    assert except_info.type is TypeError

    # array value in argument
    with pytest.raises(Exception) as except_info:
        MCMC_chain(
            10,
            5,
            sd=1,
            mean=np.array([5]),
            variance=0.5,
        )
    assert except_info.type is TypeError


def test_adaptative_sd():

    x = MCMC_chain(
        10,
        5,
        sd=1,
        mean=np.array([5]),
        variance=np.array([0.5]),
    )

    assert x.adapt_sd() is None
    assert len(x.sd()) == 1

    x.adaptative_sd(True)
    assert x.adapt_sd() is None

    assert len(x.sd()) == 2


def test_gibbs_sampler_step():
    x = MCMC_chain(
        10,
        5,
        sd=0.5,
        mean=np.array([5]),
        variance=np.array([0.1]),
    )

    for i in range(100):
        x.gibbs_sampler_step(lambda i, x: 0, None)

    chain_mean = np.mean(x.chain(), axis=1)
    assert np.mean(np.abs(chain_mean[60:] / 5 - 1)) < 0.05

    a_rate = x.acceptance_rate()
    assert np.abs(np.mean(a_rate[60:]) - 0.55) < 0.1

    assert len(x.sd()) == 1

    x.adaptative_sd(True)
    x.gibbs_sampler_step(lambda i, x: 0, None)
    assert len(x.sd()) == 2


test_gibbs_sampler_step()
