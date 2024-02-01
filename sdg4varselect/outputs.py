"""
Module for results handling objects.

Create by antoine.caillebotte@inrae.fr
"""
# pylint: disable=C0116
from collections import namedtuple

import gzip
import pickle


from sdg4varselect.models.abstract_joint_model import AbstractJointModel

import jax.numpy as jnp


class sdg4vsResults:
    def __new__(self, *args, **kwargs):
        # self._model_cst = {"N": N, "J": J, "P": P}
        return super(sdg4vsResults, self).__new__(self, *args, **kwargs)

    # def __init__(self, *args, N=None,J=None,P=None, **kwargs):
    #     if model is not None:
    #         self.model_cst = {"N": model.N, "J": model.J, "P": model.DIM_HD}

    # @property
    # def model_cst(self):
    #     return self._model_cst

    def default_filename(self, model: type(AbstractJointModel)):
        """return filename base on model configuration"""
        return f"N{model.N}_P{model.DIM_HD}_J{model.J}"

    def save(self, model: type(AbstractJointModel), root: str, filename_default: str):
        """save the object"""
        filename = f"{filename_default}/{self.default_filename(model)}"
        pickle.dump(self, gzip.open(f"{root}/{filename}.pkl.gz", "wb"))


_SDGResults = namedtuple("SDGResults", ("theta", "FIM", "grad", "likelihood"))


class SDGResults(sdg4vsResults, _SDGResults):
    # def __new__(self, model: type(abstract_joint_model), *args, **kwargs):
    # return super(SDGResults, self).__new__(self, model, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        pass
        # sdg4vsResults.__init__(self, *args, **kwargs)
        # _SDGResults.__init__(self, *args, **kwargs)

    @classmethod
    def new_from_list(cls, sdg_res):
        # assert isinstance(sdg_res, list)
        # assert len(sdg_res) != 0
        # assert isinstance(sdg_res[0], list)
        # print(type(sdg_res))
        # print(type(sdg_res[0]))

        res = [
            [sdg_res[i][j] for i in range(len(sdg_res))] for j in range(len(sdg_res[0]))
        ]

        return cls(
            theta=jnp.array(res[0]),
            FIM=res[1],
            grad=jnp.array(res[2]),
            likelihood=jnp.array(res[3]),
        )

    @property
    def last_theta(self):
        """return the last theta-array of attribut theta"""
        return self.theta[-1]

    @classmethod
    def compute_with_model(cls, prngkey, algo, model, sdg_res):
        likelihood = algo.likelihood_marginal(model, prngkey, sdg_res.last_theta)
        theta = jnp.array([model.reals1d_to_hstack_params(t) for t in sdg_res.theta])

        return cls(
            theta=theta,
            FIM=sdg_res.FIM,
            grad=sdg_res.grad,
            likelihood=likelihood,
        )

    @classmethod
    def expand_theta(cls, results, preserved_component):
        res = jnp.zeros(shape=(results.theta.shape[0], preserved_component.shape[0]))
        theta = res.at[:, jnp.where(preserved_component)].set(results.theta[:, None, :])

        return cls(
            theta=theta,
            FIM=results.FIM,
            grad=results.grad,
            likelihood=results.likelihood,
        )

    @classmethod
    def make_it_lighter(cls, results):
        return cls(
            theta=jnp.array([results.theta[0], results.theta[-1]]),
            FIM=None,
            grad=jnp.array([results.grad[0], results.grad[-1]]),
            likelihood=results.likelihood,
        )


_RegularizationPathRes = namedtuple(
    "RegularizationPathRes",
    ("listSDGResults", "bic"),
)


class RegularizationPathRes(sdg4vsResults, _RegularizationPathRes):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def make_it_lighter(cls, results):
        listSDGResults = [SDGResults.make_it_lighter(r) for r in results.listSDGResults]

        return cls(
            listSDGResults=listSDGResults,
            bic=results.bic,
        )


_variableSelectionRes = namedtuple(
    "VariableSelectionRes",
    ("listSDGResults", "theta", "regularization_path", "bic", "argmin_bic"),
)


class VariableSelectionRes(sdg4vsResults, _variableSelectionRes):
    def __init__(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    print(sdg4vsResults().default_filename(AbstractJointModel()))
    print(SDGResults(1, 2, 3, 4))

    x = SDGResults(1, 2, 3, 4)
    print(x)
    print(x.default_filename(model=AbstractJointModel()))

    print(SDGResults(FIM=1, likelihood=2, grad=3, theta=4))
