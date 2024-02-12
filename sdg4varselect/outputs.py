"""
Module for results handling objects.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116
from collections import namedtuple

import gzip
import pickle
import jax.numpy as jnp


from sdg4varselect.models.abstract_joint_model import AbstractJointModel


def _get_filename(
    model: type(AbstractJointModel),
    root: str = "",
    filename_default: str = "",
):
    """return filename"""
    filename = f"N{model.N}_P{model.DIM_HD}_J{model.J}"

    if filename_default != "":
        filename = filename_default + "_" + filename

    if root != "":
        filename = root + "/" + filename

    return filename


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

    @staticmethod
    def load(
        model: type(AbstractJointModel),
        root: str = "",
        filename_default: str = "",
    ):
        """load object"""
        filename = _get_filename(model, root, filename_default)
        out = pickle.load(gzip.open(f"{filename}.pkl.gz", "rb"))
        print(f"{filename} LOADED !")
        return out

    def save(
        self,
        model: type(AbstractJointModel),
        root: str = "",
        filename_default: str = "",
    ):
        """save the object"""
        filename = _get_filename(model, root, filename_default)
        pickle.dump(self, gzip.open(f"{filename}.pkl.gz", "wb"))
        print(f"{filename} SAVED !")


###########################################################################################################

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
    def remove_zeros(cls, results):
        preserved_component = results.last_theta != 0

        return cls(
            theta=results.theta[:, preserved_component],
            FIM=results.FIM,
            grad=results.grad[:, preserved_component],
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


###########################################################################################################

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


###########################################################################################################

_variableSelectionRes = namedtuple(
    "VariableSelectionRes",
    ("listSDGResults", "theta", "regularization_path", "bic", "argmin_bic"),
)


class VariableSelectionRes(sdg4vsResults, _variableSelectionRes):
    def __init__(self, *args, **kwargs):
        pass


###########################################################################################################

_MultiRunRes = namedtuple(
    "MultiRunRes",
    ("MultiRun", "lbd_set", "chrono", "N", "J", "P", "C"),
)


class MultiRunRes(sdg4vsResults, _MultiRunRes):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def new_from_model(cls, multi_run, lbd_set, chrono, model, censoring_rate):
        return cls(
            MultiRun=multi_run,
            lbd_set=lbd_set,
            chrono=chrono,
            N=model.N,
            J=model.J,
            P=model.DIM_HD,
            C=censoring_rate,
        )

    @classmethod
    def new_from_list(cls, multi_res):
        return cls(
            MultiRun=[res.MultiRun for res in multi_res],
            lbd_set=multi_res[0].lbd_set,
            chrono=[res.chrono for res in multi_res],
            N=[res.N for res in multi_res],
            J=[res.J for res in multi_res],
            P=[res.P for res in multi_res],
            C=[res.C for res in multi_res],
        )


if __name__ == "__main__":
    myModel = AbstractJointModel()
    print(sdg4vsResults())
    print(SDGResults(1, 2, 3, 4))

    x = SDGResults(1, 2, 3, 4)
    print(x)

    print(SDGResults(FIM=1, likelihood=2, grad=3, theta=4))

    x.save(myModel, root="", filename_default="x")
    print(sdg4vsResults.load(myModel, filename_default="x"))

    # A = namedtuple("A",("a"))
    # B = namedtuple("B",("b"))

    # class C(A,B):
    #     def __init__(self, *args, **kwargs):
    #         pass

    # print(C(a = 1, b = 2))
