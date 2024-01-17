import pickle
import gzip


from sdg4varselect.logistic import Logistic_JM
from sdg4varselect.algo import estim_res

N = 100
model = Logistic_JM(N=N, J=5, DIM_HD=10)
filename = f"testN_s{0}_N{model.N}_P{model.DIM_HD}_J{model.J}"

R = pickle.load(gzip.open(f"files/{filename}.pkl.gz", "rb"))


c = 0
nrun = 0

for c in range(len(R["res"])):
    for nrun in range(len(R["res"][nrun])):
        reg_path = R["res"][c][nrun].regularization_path
        for i in range(len(reg_path)):
            R["res"][c][nrun].regularization_path[i] = estim_res(
                theta=R["res"][c][nrun].regularization_path[i].theta[-1],
                FIM=None,
                grad=None,
                likelihood=R["res"][c][nrun].regularization_path[i].likelihood,
            )

# _C{int(jnp.array(lcensoring_rate).mean()*100)}"
pickle.dump(R, gzip.open(f"files/testN_{filename}_small.pkl.gz", "wb"))
print(f"{filename} SAVED !")
