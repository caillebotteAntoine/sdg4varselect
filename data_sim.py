import numpy as np
from sdg4varselect.solver import solver, solver_init
from sdg4varselect.parameter import par_noise_variance


# === Data simulation === #

theta_star = {
    "beta1": np.array([200]),
    "beta2": np.array([500]),
    "beta3": np.array([150.0]),
    "gamma2_1": np.array([40.0]),
    "gamma2_2": np.array([100.0]),
    "sigma2": np.array([100.0]),
}


def get_data(theta0, N=500, J=20):
    from sdg4varselect.logistic_model import model

    time_obs = np.linspace(100, 1500, num=J)
    phi1_obs = np.random.normal(theta_star["beta1"], np.sqrt(theta_star["gamma2_1"]), N)
    phi2_obs = np.random.normal(theta_star["beta2"], np.sqrt(theta_star["gamma2_2"]), N)
    phi3_obs = np.array([theta_star["beta3"] for i in range(N)])
    eps = np.random.normal(0, np.sqrt(theta_star["sigma2"]), (N, J))

    Y_obs = model(time_obs, phi1_obs, phi2_obs, phi3_obs) + eps

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
    s.add_parameter(par_noise_variance(theta0["sigma2"], "Y", model, "sigma2"))
    s.set_data("Y", "time", "phi1", "phi2", "phi3")
    s.init_parameters()

    return s
