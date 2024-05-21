# pylint: disable=C0114

import sys
from sdg4varselect.outputs import MultiRunRes


model_name = sys.argv[1]
i_min = int(sys.argv[2])
i_max = int(sys.argv[3])
N = int(sys.argv[4])
P = int(sys.argv[5])


class M:
    """model"""

    def __init__(self, n, j, p, **kwargs):
        self.N = n
        self.J = j
        self.P = p

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"{model_name}_N{self.N}_J{self.J}_P{self.P}"


model = M(n=N, j=15, p=P)


def read(s):
    """read files results for n and p as parameter"""
    r = MultiRunRes.load(model, "files_unmerged", f"S{s}")
    r.make_it_lighter()
    return r


def read_multi_files(S):
    """read multiple files results"""
    out = [read(s=S[0])]
    for s in S[1:]:
        out.append(read(s=s))

    MultiRunRes(out).save(model, root="", filename_add_on=f"S({S[0]}, {S[-1]})")

    return out


_ = read_multi_files(S=[i for i in range(i_min, i_max)])
