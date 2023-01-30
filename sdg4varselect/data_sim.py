import numpy as np
from solver import solver, solver_init
from parameter import par_noise_variance


# === Data simulation === #
def logistic_curve(x, supremum: float, midpoint: float, growth_rate: float) -> float:
    return supremum / (1 + np.exp(-(x - midpoint) / growth_rate))


def nlmem(time, phi1, phi2, phi3, **kwargs):
    N = len(phi1)
    out = [logistic_curve(time, phi1[i], phi2[i], phi3[i]) for i in range(N)]
    return np.array(out)


def loglikelihood(i: int, theta, Y, time, phi1, phi2, phi3) -> float:
    pred = logistic_curve(time, phi1[i], phi2[i], phi3[i])
    out = sum(pow(Y[i] - pred, 2))
    return -out / (2 * theta.sigma2)


theta_star = {
    "beta1": np.array([200]),
    "beta2": np.array([500]),
    "beta3": np.array([150.0]),
    "gamma2_1": np.array([40.0]),
    "gamma2_2": np.array([100.0]),
    "sigma2": np.array([100.0]),
}


def get_data(theta0, N=500, J=20):

    time_obs = np.linspace(100, 1500, num=J)
    phi1_obs = np.random.normal(theta_star["beta1"], np.sqrt(theta_star["gamma2_1"]), N)
    phi2_obs = np.random.normal(theta_star["beta2"], np.sqrt(theta_star["gamma2_2"]), N)
    phi3_obs = np.array([theta_star["beta3"] for i in range(N)])
    eps = np.random.normal(0, np.sqrt(theta_star["sigma2"]), (N, J))

    Y_obs = nlmem(time_obs, phi1_obs, phi2_obs, phi3_obs) + eps

    # === solver init === #
    s = solver_init(
        solver(),
        theta0,
        mean_name="beta",
        variance_name="gamma2_",
        mcmc_name="phi",
        dim={"phi1": N, "phi2": N, "phi3": N},
        sd={"phi1": 20, "phi2": 20, "phi3": 10},
    )

    s.add_variable("Y", Y_obs)
    s.add_variable("time", time_obs)
    s.add_parameter(par_noise_variance(theta0["sigma2"], "Y", nlmem, "sigma2"))
    s.set_data("Y", "time", "phi1", "phi2", "phi3")
    s.init_parameters()

    return s
