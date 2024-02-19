# pylint: disable=C0114
from sdg4varselect.models.wcox_mem_joint_model import (
    create_logistic_weibull_jm,
)
from sdg4varselect.outputs import MultiRegRes


def read(n, p, c, s=2):
    """read files results for n and p as parameter"""
    model = create_logistic_weibull_jm(N=n, J=5, P=p)
    # config = {"N": n, "J": 5, "P": p, "C": c}
    return MultiRegRes.load(model, "", f"C{c}_S{s}")


def read_multi_files(N, P, C, S):
    """read multiple files results"""
    model = create_logistic_weibull_jm(N=N, J=5, P=P)
    out = read(N, P, C, s=S[0])
    for s in S[1:]:
        out += read(N, P, C, s=s)

    out.save(model, root="../files", filename_add_on=f"C{C}_S({S[0]}, {S[-1]})")

    return out


N = 100
P = 10
C = 0


read_multi_files(N, P, C, S=(0, 1, 2))
