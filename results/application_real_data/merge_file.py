# pylint: disable=C0114

import sys
from sdg4varselect.outputs import MultiRunRes


class M:
    """model"""

    def __init__(self, **kwargs):
        self.N = 220
        self.J = 18
        self.P = 5

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"LogisticMEM_N{self.N}_J{self.J}_P{self.P}"


model = M()


def read(s):
    """read files results for n and p as parameter"""
    r = MultiRunRes.load(model, f"files_unmerged", f"S{s}")
    r.make_it_lighter()
    return r


def read_multi_files(S):
    """read multiple files results"""
    out = [read(s=S[0])]
    for s in S[1:]:
        out.append(read(s=s))

    MultiRunRes(out).save(model, root=f"files", filename_add_on=f"S({S[0]}, {S[-1]})")

    return out


_ = read_multi_files(S=[i for i in range(0, 9)])
