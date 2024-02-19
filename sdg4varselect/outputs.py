"""
Module for results handling objects.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, E1133

import gzip
import pickle
from datetime import timedelta
import jax.numpy as jnp

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel


def autoproperty(attribute_name: str):
    attribute_name = "_" + attribute_name

    def getter(self):
        return getattr(self, attribute_name)

    return property(getter, None)


class IsIterable:
    """define a object that can be iterable"""

    def __init__(self, name: str, results: list):
        self._name = name
        self.__dict__[self._name] = results

    def __len__(self):
        return len(self.__dict__[self._name])

    def __getitem__(self, i):
        return self.__dict__[self._name][i]

    def __iter__(self):
        for res in self.__dict__[self._name]:
            yield res


class HasLastTheta:
    """define a object that has a last theta methode"""

    def __init__(self):
        pass

    @property
    def last_theta(self):
        return [x.last_theta for x in self]


class HasLikelihood:
    """define a object that has a likelihood methode"""

    def __init__(self):
        pass

    @property
    def likelihood(self):
        return jnp.array([x.likelihood for x in self])


class CanBeLightened:
    """define a object that has can be lightened"""

    def __init__(self):
        pass

    def make_it_lighter(self):
        for res in self:
            res.make_it_lighter()


class HasChrono:
    """define a object that has chrono property"""

    def __init__(self, list_with_chrono: list):
        self._chrono = timedelta()

        for item in list_with_chrono:
            self._chrono += item.chrono

    chrono = autoproperty("chrono")


class GDResultsHandler(CanBeLightened, HasLastTheta, HasLikelihood):
    """define a object that handle GDResults"""

    def __init__(self):
        CanBeLightened.__init__(self)
        HasLastTheta.__init__(self)
        HasLikelihood.__init__(self)


def _get_filename(
    model: type[AbstractModel],
    root: str = "",
    filename_add_on: str = "",
):
    """return filename"""
    filename = model.name

    if filename_add_on != "":
        filename = filename + "_" + filename_add_on

    if root != "":
        filename = root + "/" + filename

    return filename


class sdg4vsResults:

    @staticmethod
    def load(
        model: type[AbstractModel],
        root: str = "",
        filename_add_on: str = "",
    ):
        """load object"""
        filename = _get_filename(model, root, filename_add_on)
        out = pickle.load(gzip.open(f"{filename}.pkl.gz", "rb"))
        print(f"{filename} LOADED !")
        return out

    def save(
        self,
        model: type[AbstractModel],
        root: str = "",
        filename_add_on: str = "",
    ):
        """save the object"""
        filename = _get_filename(model, root, filename_add_on)
        pickle.dump(self, gzip.open(f"{filename}.pkl.gz", "wb"))
        print(f"{filename} SAVED !")


###########################################################################################################


class GDResults(sdg4vsResults):

    def __init__(self, theta, FIM, grad, chrono=0, likelihood=jnp.nan, **kwargs):
        self._theta = theta
        self._FIM = FIM
        self._grad = grad
        self._chrono = chrono
        self._likelihood = likelihood

    theta = autoproperty("theta")
    FIM = autoproperty("FIM")
    grad = autoproperty("grad")
    chrono = autoproperty("chrono")
    likelihood = autoproperty("likelihood")

    @classmethod
    def new_from_list(cls, sdg_res, chrono):
        res = [
            [sdg_res[i][j] for i in range(len(sdg_res))] for j in range(len(sdg_res[0]))
        ]

        return cls(
            theta=jnp.array(res[0]),
            FIM=res[1],
            grad=jnp.array(res[2]),
            chrono=chrono,
            likelihood=jnp.nan,
        )

    @property
    def last_theta(self):
        """return the last theta-array of attribut theta"""
        return self._theta[-1]

    @classmethod
    def compute_with_model(cls, model, sdg_res, likelihood=None) -> "GDResults":
        # likelihood = algo.likelihood_marginal(model, sdg_res.last_theta)
        theta = jnp.array([model.reals1d_to_hstack_params(t) for t in sdg_res.theta])

        return cls(
            theta=theta,
            FIM=sdg_res.FIM,
            grad=sdg_res.grad,
            chrono=sdg_res.chrono,
            likelihood=likelihood if likelihood is not None else sdg_res.likelihood,
        )

    @classmethod
    def expand_theta(cls, results, preserved_component) -> "GDResults":
        res = jnp.zeros(shape=(results.theta.shape[0], preserved_component.shape[0]))
        theta = res.at[:, jnp.where(preserved_component)].set(results.theta[:, None, :])

        return cls(
            theta=theta,
            FIM=results.FIM,
            grad=results.grad,
            chrono=results.chrono,
            likelihood=results.likelihood,
        )

    @classmethod
    def remove_zeros(cls, results) -> "GDResults":
        preserved_component = results.last_theta != 0

        return cls(
            theta=results.theta[:, preserved_component],
            FIM=results.FIM,
            grad=results.grad[:, preserved_component],
            chrono=results.chrono,
            likelihood=results.likelihood,
        )

    @classmethod
    def make_it_lighter(cls, results) -> "GDResults":
        return cls(
            theta=jnp.array([results.theta[0], results.theta[-1]]),
            FIM=None,
            grad=jnp.array([results.grad[0], results.grad[-1]]),
            chrono=results.chrono,
            likelihood=results.likelihood,
        )


###########################################################################################################


class MultiRunRes(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
    def __init__(self, multi_run: list):
        GDResultsHandler.__init__(self)
        while sdg4vsNanError in multi_run:
            multi_run.remove(sdg4vsNanError)

        HasChrono.__init__(self, multi_run)
        IsIterable.__init__(self, "multi_run", multi_run)

    @property
    def nrun(self):
        """return number of run"""
        return len(self)


###########################################################################################################


class RegularizationPathRes(sdg4vsResults, IsIterable, HasChrono, CanBeLightened):
    def __init__(self, multi_run: MultiRunRes, bic, argmin_bic, lbd_set):
        HasChrono.__init__(self, multi_run)
        IsIterable.__init__(self, "multi_run", multi_run)

        self._bic = bic
        self._argmin_bic = argmin_bic
        self._lbd_set = lbd_set

    bic = autoproperty("bic")
    argmin_bic = autoproperty("argmin_bic")
    lbd_set = autoproperty("lbd_set")

    @property
    def final_result(self):
        return self[self._argmin_bic]

    @classmethod
    def switch_runs(cls, results):
        nrun = len(results.multi_run)
        nresolution = len(results.multi_run[0])
        runs = [
            MultiRunRes([results.multi_run[i][j] for i in range(nrun)])
            for j in range(nresolution)
        ]

        return cls(
            multi_run=runs,
            bic=results.bic,
            argmin_bic=results.argmin_bic,
            lbd_set=results.lbd_set,
        )

    @property
    def last_theta(self):
        return self[self.argmin_bic].last_theta


###########################################################################################################


class MultiRegRes(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
    def __init__(self, multi_run: list[RegularizationPathRes]):
        GDResultsHandler.__init__(self)
        HasChrono.__init__(self, multi_run)
        IsIterable.__init__(self, "multi_run", multi_run)

    @property
    def nrun(self):
        """return number of run"""
        return len(self)


###########################################################################################################


class TestResults(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
    def __init__(self, tests: list[MultiRegRes], test_config: list[dict]):
        GDResultsHandler.__init__(self)
        HasChrono.__init__(self, tests)
        IsIterable.__init__(self, "tests", tests)
        assert len(tests) == len(test_config)

        self._config = test_config

    config = autoproperty("config")

    @property
    def ntest(self):
        """return number of run"""
        return len(self)

    def get_scenarios_labels(self, key: str) -> list[str]:
        return [f"{key} = {c[key]}" for c in self.config]
