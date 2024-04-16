# pylint: disable=C0114

from sdg4varselect.outputs import MultiRegRes

from sdg4varselect.models import AbstractModel


class m:
    """model"""

    def __init__(self, n, j, p, **kwargs):
        self.N = n
        self.J = j
        self.P = p

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"HDLogisticMEM_N{self.N}_J{self.J}_P{self.P}"


model = m(n=1000, j=10, p=50)


def read(s):
    """read files results for n and p as parameter"""
    r = MultiRegRes.load(model, "", f"_S{s}")
    r.make_it_lighter()
    return r


def read_multi_files(S):
    """read multiple files results"""
    out = read(s=S[0])
    for s in S[1:]:
        out += read(s=s)

    out.save(model, root="", filename_add_on=f"S({S[0]}, {S[-1]})")

    return out


read_multi_files(S=[i + 1 for i in range(10)])
