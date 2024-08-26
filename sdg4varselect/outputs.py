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
            fim=res[2],
            grad=jnp.array(res[1]),
            chrono=chrono,
            likelihood=jnp.nan,
        )

    @property
    def last_theta(self):
        """return the last theta-array of attribut theta"""
        id_not_all_nan = jnp.logical_not(jnp.isnan(self.theta).all(axis=1))
        out = self.theta[id_not_all_nan][-1]
        return out

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

        self.grad = jnp.pad(
            self.grad,
            (
                (0, max_row - self.grad.shape[0]),
                (0, 0 if max_col is None else (max_col - self.grad.shape[1])),
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
    ebic: jnp.ndarray = jnp.nan
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

    def standardize(self):
        multi_run = []
        bic = []
        k = 0

        while k < len(self):
            supp_k = self[k].last_theta[-1] != 0
            same_supp = []

            i = k
            supp_i = supp_k
            while i < len(self) and (supp_i == supp_k).all():
                same_supp.append(i)
                i += 1
                if i < len(self):
                    supp_i = self[i].last_theta[-1] != 0

            best_supp_id = k + self.bic[-1, same_supp].argmin()
            k = i
            # print(same_supp, best_supp_id)
            for _ in same_supp:
                multi_run.append(self[best_supp_id])
                bic.append(self.bic[:, best_supp_id])

        bic = jnp.array(bic).T
        return RegularizationPathRes(multi_run, bic, bic[-1].argmin(), self.lbd_set)


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

    def __init__(self, tests: list, test_config: list[dict]):
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

        if len(tests) > 0:
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

    def sort(self, key: str):
        assert key in self.config[0]
        id = sorted([i for i in range(len(self))], key=lambda x: self.config[x][key])

        self.tests = [self.tests[i] for i in id]
        self.config = [self.config[i] for i in id]

        return self

    def filter(
        self,
        keep_type=False,
        **kwargs,
    ):  # key: str, value: int):
        out = self
        for key, value in kwargs.items():
            assert key in out.config[0]
            id = [i for i in range(len(out)) if out.config[i][key] == value]
            out = TestResults([out.tests[i] for i in id], [out.config[i] for i in id])

        if not keep_type and len(out) == 1:
            return out[0]
        return out


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
