"""
Module that define functions to perform a selection and estimation.

Create by antoine.caillebotte@inrae.fr"""

# pylint: disable=C0116
import jax.random as jrd
import jax.numpy as jnp


from sdg4varselect.models.wcox_mem_joint_model import get_params_star

from sdg4varselect.models import WeibullCoxJM, logisticMEM
from results.logistic_model.multi_results import multi_run


import sys

seed = 0  # int(sys.argv[1])
print(seed)

lbd_set = 10 ** jnp.linspace(-2, 0, num=15)


def test(N, J, P, nrun=1, censoring=2000):

    # joint model with coxModel is all ready implement in sdg4varselect for all MixedEffectsModel
    myModel = WeibullCoxJM(logisticMEM(N=N, J=J), P=P, alpha_scale=0.001, a=800, b=10)

    p_star = myModel.new_params(
        mean_latent={"mu1": 200, "mu2": 500},
        mu3=150,
        cov_latent=jnp.diag(jnp.array([40, 100])),
        var_residual=100,
        alpha=0.005,
        beta=jnp.concatenate(  # jnp.zeros(shape=(myModel.P,)),  #
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(myModel.P - 4,))]
        ),
    )
    res, censoring_rate = multi_run(
        jrd.PRNGKey(seed),
        lbd_set,
        p_star,
        myModel,
        nrun=nrun,
        censoring=censoring,
        save_all=False,
    )

    C = "NA" if jnp.isnan(censoring_rate) else int(censoring_rate)
    res.save(myModel, root="files_unmerged", filename_add_on=f"C{C}_S{seed}")


test(1000, 5, 20)
# test(100, 5, 50)

# test(100, 5, 100)
# test(200, 5, 100)
# test(300, 5, 100)

# test(100, 5, 30)
# test(100, 5, 100)
# test(100, 5, 200)

# test(100, 5, 500)
# test(100, 5, 800)
