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


def gibbs_sampler_step():
    x = MCMC_chain(
        10,
        5,
        sd=1,
        mean=np.array([5]),
        variance=np.array([0.5]),
    )


x = MCMC_chain(
    10,
    5,
    sd=1,
    mean=np.array([5]),
    variance=np.array([0.5]),
)


for i in range(100):
    x.gibbs_sampler_step(lambda i, x: 0, None)


x.print()
