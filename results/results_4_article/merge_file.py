# pylint: disable=C0114

from sdg4varselect.outputs import MultiRunRes


class M:
    """model"""

    def __init__(self, n, j, p, **kwargs):
        self.N = n
        self.J = j
        self.P = p

    @property
    def name(self):
        """return a str called name, based on the parameter of the model"""
        return f"WCoxMemJM_N{self.N}_J{self.J}_P{self.P}"


model = M(n=200, j=15, p=10)


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

    MultiRunRes(out).save(model, root="files", filename_add_on=f"S({S[0]}, {S[-1]})")

    return out


_ = read_multi_files(S=[i + 1 for i in range(0, 1)])
