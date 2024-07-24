# pylint: disable=C0114

import sys
import os
from sdg4varselect.outputs import MultiRunRes, _get_filename

print(sys.argv)
folder = sys.argv[1]
model_name = sys.argv[2]
i_min = int(sys.argv[3])
i_max = int(sys.argv[4])
N = int(sys.argv[5])
P = int(sys.argv[6])


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
root = f"{folder}/files_unmerged"
add_on = "_fisheradagrad"

def read(s):
    """read files results for n and p as parameter"""
    r = MultiRunRes.load(model, root, f"S{s}{add_on}")
    r.make_it_lighter()
    return r


def read_multi_files(S):
    """read multiple files results"""
    out = [read(s=S[0])]
    flag = False
    for s in S[1:]:
        try:
            out.append(read(s=s))
            print(f"computation time of the {s}th file : {out[-1].chrono}")
        except:
            flag = True
            print(f"can't find the {s}th file !")

    if not flag :
        try : 
            MultiRunRes(out).save(model, root=f"{folder}/files", filename_add_on=f"S({S[0]}, {S[-1]}){add_on}")
        except:
            print("something go wrong !")
        else :
            for s in S:
                os.remove(_get_filename(model, root, f"S{s}{add_on}.pkl.gz"))
        

    return out


_ = read_multi_files(S=[i for i in range(i_min, i_max)])
