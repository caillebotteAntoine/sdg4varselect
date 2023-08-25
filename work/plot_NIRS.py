import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dt = pd.read_csv("../simulated_nirs.csv", sep=";", header=1, decimal=",")


cov = np.array(dt.values)


plt.plot(cov)


p = 100
p_max = cov.shape[0]
p_step = int(np.round(p_max / p))

cov_shrink = cov[[i * p_step for i in range(p)]]
plt.plot(cov_shrink)
