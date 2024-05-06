"""
Module for results handling objects.

Create by antoine.caillebotte@inrae.fr
"""

# pylint: disable=C0116, E1133

from dataclasses import dataclass, InitVar, field
import gzip
import pickle
from datetime import datetime, timedelta
import jax.numpy as jnp

from sdg4varselect.exceptions import sdg4vsNanError
from sdg4varselect.models.abstract.abstract_model import AbstractModel

""" qsdf"""


###########################################################################################################


class IsIterable:
    """define a object that can be iterable"""

    def __init__(self, name: str, results: list):
        self._name = name
        self.__dict__[self._name] = results

    def __len__(self):
        return len(self.__dict__[self._name])

    @property
    def iterable_data(self):
        return self.__dict__[self._name]

    def sort(self):
        self.__dict__[self._name] = sorted(
            self.__dict__[self._name],
            key=lambda x: (
                -x.likelihood if len(x.likelihood.shape) == 0 else -x.likelihood[-1]
            ),
        )

    @property
    def total_number_res(self):
        if isinstance(self.__dict__[self._name][0], IsIterable):
            return sum([iter.total_number_res for iter in self])
        return len(self)

    def __getitem__(self, i):
        return self.__dict__[self._name][i]

    def __iter__(self):
        for res in self.__dict__[self._name]:
            yield res


class HasTheta:
    """define a object that has a last theta methode"""

    def __init__(self):
        pass

    @property
    def last_theta(self):
        return jnp.array([x.last_theta for x in self])

    @property
    def theta(self):
        return jnp.array([x.theta for x in self])

    def make_it_lighter(self):
        for res in self:
            res.make_it_lighter()

    def uniform_theta(self, *args, **kwargs):
        for res in self:
            res.uniform_theta(*args, **kwargs)


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


@dataclass
class HasChrono:
    """define a object that has chrono property"""

    chrono: timedelta = timedelta()
    list_with_chrono: InitVar[list] = None

    def __init__(self, list_with_chrono: list):
        self.chrono = timedelta()
        if list_with_chrono is not None:
            for item in list_with_chrono:
                self.chrono += item.chrono


class GDResultsHandler(CanBeLightened, HasTheta, HasLikelihood):
    """define a object that handle GDResults"""

    def __init__(self):
        CanBeLightened.__init__(self)
        HasTheta.__init__(self)
        HasLikelihood.__init__(self)


###########################################################################################################


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
    """define a object that handle results for the sdg4varselect package"""

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


@dataclass
class GDResults:
    """define a object that handle gradient descent results"""

    theta: jnp.ndarray
    fim: jnp.ndarray = None
    grad: jnp.ndarray = None
    chrono: timedelta = timedelta()
    likelihood: jnp.ndarray = jnp.nan

    @classmethod
    def new_from_list(cls, sdg_res, chrono):
        res = [
            [sdg_res[i][j] for i in range(len(sdg_res))] for j in range(len(sdg_res[0]))
        ]

        return cls(
            theta=jnp.array(res[0]),
            fim=res[1],
            grad=jnp.array(res[2]),
            chrono=chrono,
            likelihood=jnp.nan,
        )

    @property
    def last_theta(self):
        """return the last theta-array of attribut theta"""
        # print(self.theta.shape)
        # print(self.theta)
        # print(jnp.isnan(self.theta).any(axis=1))
        id_not_all_nan = jnp.logical_not(jnp.isnan(self.theta).all(axis=1))
        # print(id_not_nan.shape)
        # print(id_not_nan)
        # print(self.theta[id_not_nan].shape)
        # print(self.theta[id_not_nan])
        out = self.theta[id_not_all_nan][-1]
        return out
        return out[jnp.logical_not(jnp.isnan(out))]

        return self.theta[-1][jnp.logical_not(jnp.isnan(self.theta[-1]))]
        return self.theta[jnp.logical_not(jnp.isnan(self.theta).any(axis=1))][-1]

    def reals1d_to_hstack_params(self, model):
        # likelihood = algo.likelihood_marginal(model, sdg_res.last_theta)
        self.theta = jnp.array([model.reals1d_to_hstack_params(t) for t in self.theta])

    def remove_zeros(self) -> "GDResults":
        preserved_component = self.last_theta != 0

        self.theta = self.theta[:, preserved_component]
        self.grad = self.grad[:, preserved_component]

    def make_it_lighter(self):
        id_not_nan = jnp.logical_not(jnp.isnan(self.theta).any(axis=1))
        theta = self.theta[id_not_nan]
        grad = self.grad[id_not_nan]

        self.theta = jnp.array([theta[0], theta[-1]])
        self.fim = None
        self.grad = jnp.array([grad[0], grad[-1]])

    # ========================================= #
    # ===== theta uniformization factory =====  #
    # ========================================= #
    def expand_theta(self, preserved_component):
        res = jnp.zeros(shape=(self.theta.shape[0], preserved_component.shape[0]))
        self.theta = res.at[:, jnp.where(preserved_component)].set(
            self.theta[:, None, :]
        )

    def uniform_theta(self, max_row, max_col=None):  # , max_col):
        self.theta = jnp.pad(
            self.theta,
            (
                (0, max_row - self.theta.shape[0]),
                (0, 0 if max_col is None else (max_col - self.theta.shape[1])),
            ),
            constant_values=jnp.nan,
        )


###########################################################################################################


class MultiRunRes(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
    """define a object that handle multiple GDResults"""

    def __init__(self, multi_run: list[GDResults]):
        GDResultsHandler.__init__(self)
        while sdg4vsNanError in multi_run:
            multi_run.remove(sdg4vsNanError)

        max_row = max([run.theta.shape[-2] for run in multi_run])
        # max_col = max([run.theta.shape[-1] for run in multi_run])

        for _, run in enumerate(multi_run):
            run.uniform_theta(max_row)  # , max_col)

        HasChrono.__init__(self, multi_run)
        IsIterable.__init__(self, "multi_run", multi_run)

    @property
    def nrun(self):
        """return number of run"""
        return len(self)

    def __add__(self, res):
        return MultiRunRes(self.iterable_data + res.iterable_data)

    def reduce_number_run(self, keep=None, keep_percentage=None):
        if keep_percentage is not None:
            keep = int(len(self) * keep_percentage)

        assert isinstance(keep, int)
        return MultiRunRes(self[:keep])


###########################################################################################################

if __name__ == "__main__":
    s = datetime.now()
    x = GDResults(jnp.array([[1.0, 2], [3, 4], [5, 6]]), chrono=s - datetime.now())
    y = GDResults(2 * jnp.array([[1, 2.1], [3, 4]]), chrono=s - datetime.now())

    r = MultiRunRes([x, y])
    print(r.theta)
    print(r.last_theta)
    print("add", (r + r).theta)

    a = GDResults(jnp.array([[1.0, 2]]), chrono=s - datetime.now())
    b = GDResults(2 * jnp.array([[1, 2.1]]), chrono=s - datetime.now())

    m = MultiRunRes([r, MultiRunRes([a, b])])
    print(m.theta)
    print(m.last_theta)
    print("add", (m + m).theta)

###########################################################################################################


@dataclass
class RegularizationPathRes(MultiRunRes):
    """define a object that handle regularization path results"""

    bic: jnp.ndarray = jnp.nan
    argmin_bic: int = None
    lbd_set: jnp.ndarray = jnp.nan

    def __init__(self, multi_run: MultiRunRes, bic, argmin_bic, lbd_set):
        MultiRunRes.__init__(self, multi_run)

        self.bic = bic
        self.argmin_bic = argmin_bic
        self.lbd_set = lbd_set

    @property
    def final_result(self):
        return self[self.argmin_bic]

    @property
    def theta(self):
        return self.final_result.theta

    @property
    def last_theta(self):
        return self.final_result.last_theta

    @property
    def likelihood(self):
        return self.final_result.likelihood


###########################################################################################################

if __name__ == "__main__":
    s = datetime.now()
    x = GDResults(jnp.array([[1.0, 2], [3, 4], [5, 6]]), chrono=s - datetime.now())
    y = GDResults(2 * jnp.array([[1, 2.1], [3, 4]]), chrono=s - datetime.now())

    r = MultiRunRes([x, y])
    print(r.theta)
    print(r.last_theta)
    print("add", (r + r).theta)

    a = GDResults(jnp.array([[1.0, 2]]), chrono=s - datetime.now())
    b = GDResults(2 * jnp.array([[1, 2.1]]), chrono=s - datetime.now())

    m = RegularizationPathRes([r, MultiRunRes([a, b])], 0, 0, 0)
    print(m.theta)
    print(m.last_theta)
    print("add", (m + m).theta)


###########################################################################################################


@dataclass
class TestResults(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
    """define a object that handle testing results"""

    config: list[dict] = field(default_factory=list)

    def __init__(self, tests: list[MultiRunRes], test_config: list[dict]):
        # max_col_theta = jnp.array(
        #     [
        #         [[run.theta.shape[1] for run in res] for res in t]
        #         for t in tests
        #     ]
        # ).max()
        # for t in tests:
        #     for res in t:
        #         for run in res.final_result:
        #             run.uniform_theta(max_col_theta)

        max_row = max([run.theta.shape[-2] for run in tests])
        max_col = max([run.theta.shape[-1] for run in tests])

        for _, run in enumerate(tests):
            run.uniform_theta(max_row, max_col)

        GDResultsHandler.__init__(self)
        HasChrono.__init__(self, tests)
        IsIterable.__init__(self, "tests", tests)
        assert len(tests) == len(test_config)

        self.config = test_config

    @property
    def ntest(self):
        """return number of run"""
        return len(self)

    def get_scenarios_labels(self, key: str) -> list[str]:
        return [f"{key} = {c[key]}" for c in self.config]


###########################################################################################################


def get_all_theta(output):
    if isinstance(output, RegularizationPathRes):
        return get_all_theta(output.final_result)
    elif isinstance(output, MultiRunRes):
        out = [res.theta for res in output]
    elif isinstance(output, GDResults):
        out = [output.theta]
    else:
        raise TypeError(
            "output must be RegularizationPathRes, MultiRunRes or GDResults"
        )

    return jnp.array(out).T  # dim_standardize(out)


# def dim_standardize(list_x: list):
#     max_row = max([x.shape[0] for x in list_x])
#     max_col = max([x.shape[1] for x in list_x])
#     return jnp.array(
#         [
#             jnp.pad(
#                 x,
#                 ((0, max_row - x.shape[0]), (0, max_col - x.shape[1])),
#                 constant_values=jnp.nan,
#             )
#             for x in list_x
#         ]
#     ).T


# def autoproperty(attribute_name: str):
#     attribute_name = "_" + attribute_name

#     def getter(self):
#         return getattr(self, attribute_name)

#     return property(getter, None)


# class IsIterable:
#     """define a object that can be iterable"""

#     def __init__(self, name: str, results: list):
#         self._name = name
#         self.__dict__[self._name] = results

#     def __len__(self):
#         return len(self.__dict__[self._name])

#     @property
#     def iterable_data(self):
#         return self.__dict__[self._name]

#     def sort(self):
#         self.__dict__[self._name] = sorted(
#             self.__dict__[self._name],
#             key=lambda x: (
#                 x.likelihood if len(x.likelihood.shape) == 0 else x.likelihood[-1]
#             ),
#         )

#     @property
#     def total_number_res(self):
#         if isinstance(self.__dict__[self._name][0], IsIterable):
#             return sum([iter.total_number_res for iter in self])
#         return len(self)

#     def __getitem__(self, i):
#         return self.__dict__[self._name][i]

#     def __iter__(self):
#         for res in self.__dict__[self._name]:
#             yield res


# class HasTheta:
#     """define a object that has a last theta methode"""

#     def __init__(self):
#         pass

#     @property
#     def last_theta(self):
#         return jnp.array([x.last_theta for x in self])

#     @property
#     def theta(self):
#         return jnp.array([x.theta for x in self])

#     def make_it_lighter(self):
#         for res in self:
#             res.make_it_lighter()

#     def expand_iteration(self, max_row):
#         for res in self:
#             res.expand_iteration(max_row)

#     def uniform_theta(self, max_col):
#         for res in self:
#             res.uniform_theta(max_col)


# class HasLikelihood:
#     """define a object that has a likelihood methode"""

#     def __init__(self):
#         pass

#     @property
#     def likelihood(self):
#         return jnp.array([x.likelihood for x in self])


# class CanBeLightened:
#     """define a object that has can be lightened"""

#     def __init__(self):
#         pass

#     def make_it_lighter(self):
#         for res in self:
#             res.make_it_lighter()


# class HasChrono:
#     """define a object that has chrono property"""

#     def __init__(self, list_with_chrono: list):
#         self._chrono = timedelta()

#         for item in list_with_chrono:
#             self._chrono += item.chrono

#     chrono = autoproperty("chrono")


# class GDResultsHandler(CanBeLightened, HasTheta, HasLikelihood):
#     """define a object that handle GDResults"""

#     def __init__(self):
#         CanBeLightened.__init__(self)
#         HasTheta.__init__(self)
#         HasLikelihood.__init__(self)


# def _get_filename(
#     model: type[AbstractModel],
#     root: str = "",
#     filename_add_on: str = "",
# ):
#     """return filename"""
#     filename = model.name

#     if filename_add_on != "":
#         filename = filename + "_" + filename_add_on

#     if root != "":
#         filename = root + "/" + filename

#     return filename


# class sdg4vsResults:

#     @staticmethod
#     def load(
#         model: type[AbstractModel],
#         root: str = "",
#         filename_add_on: str = "",
#     ):
#         """load object"""
#         filename = _get_filename(model, root, filename_add_on)
#         out = pickle.load(gzip.open(f"{filename}.pkl.gz", "rb"))
#         print(f"{filename} LOADED !")
#         return out

#     def save(
#         self,
#         model: type[AbstractModel],
#         root: str = "",
#         filename_add_on: str = "",
#     ):
#         """save the object"""
#         filename = _get_filename(model, root, filename_add_on)
#         pickle.dump(self, gzip.open(f"{filename}.pkl.gz", "wb"))
#         print(f"{filename} SAVED !")


# ###########################################################################################################


# class GDResults(sdg4vsResults):

#     def __init__(self, theta, FIM, grad, chrono=0, likelihood=jnp.nan, **kwargs):
#         self._theta = theta
#         self._FIM = FIM
#         self._grad = grad
#         self._chrono = chrono
#         self._likelihood = likelihood

#     theta = autoproperty("theta")
#     FIM = autoproperty("FIM")
#     grad = autoproperty("grad")
#     chrono = autoproperty("chrono")
#     likelihood = autoproperty("likelihood")

#     @classmethod
#     def new_from_list(cls, sdg_res, chrono):
#         res = [
#             [sdg_res[i][j] for i in range(len(sdg_res))] for j in range(len(sdg_res[0]))
#         ]

#         return cls(
#             theta=jnp.array(res[0]),
#             FIM=res[1],
#             grad=jnp.array(res[2]),
#             chrono=chrono,
#             likelihood=jnp.nan,
#         )

#     @property
#     def last_theta(self):
#         """return the last theta-array of attribut theta"""
#         return self._theta[jnp.logical_not(jnp.isnan(self._theta).any(axis=1))][-1]

#     @classmethod
#     def compute_with_model(
#         cls, model, sdg_res, likelihood=None, chrono=None
#     ) -> "GDResults":
#         # likelihood = algo.likelihood_marginal(model, sdg_res.last_theta)
#         chrono_start = datetime.now()
#         theta = jnp.array([model.reals1d_to_hstack_params(t) for t in sdg_res.theta])

#         return cls(
#             theta=theta,
#             FIM=sdg_res.FIM,
#             grad=sdg_res.grad,
#             chrono=(sdg_res.chrono if chrono is None else chrono)
#             + datetime.now()
#             - chrono_start,
#             likelihood=likelihood if likelihood is not None else sdg_res.likelihood,
#         )

#     @classmethod
#     def expand_theta(cls, results, preserved_component) -> "GDResults":
#         res = jnp.zeros(shape=(results.theta.shape[0], preserved_component.shape[0]))
#         theta = res.at[:, jnp.where(preserved_component)].set(results.theta[:, None, :])

#         return cls(
#             theta=theta,
#             FIM=results.FIM,
#             grad=results.grad,
#             chrono=results.chrono,
#             likelihood=results.likelihood,
#         )

#     @classmethod
#     def remove_zeros(cls, results) -> "GDResults":
#         preserved_component = results.last_theta != 0

#         return cls(
#             theta=results.theta[:, preserved_component],
#             FIM=results.FIM,
#             grad=results.grad[:, preserved_component],
#             chrono=results.chrono,
#             likelihood=results.likelihood,
#         )

#     def make_it_lighter(self):
#         self._theta = jnp.array([self.theta[0], self.theta[-1]])
#         self._FIM = None
#         self._grad = jnp.array([self.grad[0], self.grad[-1]])
#         self._chrono = self.chrono
#         self._likelihood = self.likelihood

#     def expand_iteration(self, max_row):
#         self._theta = jnp.pad(
#             self._theta,
#             ((0, max_row - self._theta.shape[0]), (0, 0)),
#             constant_values=jnp.nan,
#         )

#     def uniform_theta(self, max_col):
#         self._theta = jnp.pad(
#             self._theta,
#             ((0, 0), (0, max_col - self._theta.shape[1])),
#             constant_values=jnp.nan,
#         )


# ###########################################################################################################


# class MultiRunRes(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
#     def __init__(self, multi_run: list[GDResults]):
#         GDResultsHandler.__init__(self)
#         while sdg4vsNanError in multi_run:
#             multi_run.remove(sdg4vsNanError)

#         max_iteration = max([run.theta.shape[0] for run in multi_run])
#         for _, run in enumerate(multi_run):
#             run.expand_iteration(max_iteration)

#         HasChrono.__init__(self, multi_run)
#         IsIterable.__init__(self, "multi_run", multi_run)

#     @property
#     def nrun(self):
#         """return number of run"""
#         return len(self)


# ###########################################################################################################


# class RegularizationPathRes(sdg4vsResults, IsIterable, HasChrono, CanBeLightened):
#     def __init__(self, multi_run: MultiRunRes, bic, argmin_bic, lbd_set):
#         HasChrono.__init__(self, multi_run)
#         IsIterable.__init__(self, "multi_run", multi_run)

#         self._bic = bic
#         self._argmin_bic = argmin_bic
#         self._lbd_set = lbd_set

#     bic = autoproperty("bic")
#     argmin_bic = autoproperty("argmin_bic")
#     lbd_set = autoproperty("lbd_set")

#     @property
#     def final_result(self):
#         return self[self._argmin_bic]

#     @classmethod
#     def switch_runs(cls, results):
#         nrun = len(results.multi_run)
#         nresolution = len(results.multi_run[0])
#         runs = [
#             MultiRunRes([results.multi_run[i][j] for i in range(nrun)])
#             for j in range(nresolution)
#         ]

#         return cls(
#             multi_run=runs,
#             bic=results.bic,
#             argmin_bic=results.argmin_bic,
#             lbd_set=results.lbd_set,
#         )

#     @property
#     def last_theta(self):
#         return self[self.argmin_bic].last_theta

#     @property
#     def likelihood(self):
#         return self[self.argmin_bic].likelihood


# ###########################################################################################################


# class MultiRegRes(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
#     def __init__(self, multi_run: list[RegularizationPathRes]):
#         GDResultsHandler.__init__(self)
#         HasChrono.__init__(self, multi_run)
#         IsIterable.__init__(self, "multi_run", multi_run)

#     @property
#     def nrun(self):
#         """return number of run"""
#         return len(self)

#     def __add__(self, res):
#         return MultiRegRes(self.iterable_data + res.iterable_data)


# ###########################################################################################################


# class TestResults(sdg4vsResults, IsIterable, HasChrono, GDResultsHandler):
#     def __init__(self, tests: list[MultiRegRes], test_config: list[dict]):
#         max_col_theta = jnp.array(
#             [
#                 [[run.theta.shape[1] for run in res.final_result] for res in t]
#                 for t in tests
#             ]
#         ).max()
#         for t in tests:
#             for res in t:
#                 for run in res.final_result:
#                     run.uniform_theta(max_col_theta)

#         GDResultsHandler.__init__(self)
#         HasChrono.__init__(self, tests)
#         IsIterable.__init__(self, "tests", tests)
#         assert len(tests) == len(test_config)

#         self._config = test_config

#     config = autoproperty("config")

#     @property
#     def ntest(self):
#         """return number of run"""
#         return len(self)

#     def get_scenarios_labels(self, key: str) -> list[str]:
#         return [f"{key} = {c[key]}" for c in self.config]


# ###########################################################################################################


# def get_all_theta(output):
#     if isinstance(output, RegularizationPathRes):
#         return get_all_theta(output.final_result)
#     elif isinstance(output, MultiRunRes):
#         out = [res.theta for res in output]
#     elif isinstance(output, GDResults):
#         out = [output.theta]
#     else:
#         raise TypeError(
#             "output must be RegularizationPathRes, MultiRunRes or GDResults"
#         )

#     return jnp.array(out).T  # dim_standardize(out)
