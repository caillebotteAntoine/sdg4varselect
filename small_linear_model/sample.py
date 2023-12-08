import numpy as np
import parametrization_cookbook.jax as pc
from sdg4varselect.data_generation import mem_simulation, cox_simulation, cov_simulation

from sdg4varselect import Gradient, jrd, jnp
from collections import namedtuple


from sdg4varselect.linear_model import (
    linear_curve,
    linear_curve_float,
)

# ============================================================= #
# ====================== PARAMIETRIZATION ===================== #
# ============================================================= #
params_weibull = namedtuple(
    "params_weibull",
    ("mu1", "mu2", "gamma2_1", "gamma2_2", "sigma2", "a", "b", "alpha", "beta"),
)


# ===================================================== #
# ================== DATA GENERATION ================== #
# ===================================================== #
def nlmem_simulation(params, PRNGKey, N_IND, J_OBS, t_min, t_max):
    """return longitudinal simulation
    and latente variable simulation in the two dict"""

    def linear_fct(time, phi1, phi2):
        return linear_curve(
            time=time,
            intercept=phi1,
            slope=phi2,
        )

    random_effects = {"phi1": ("mu1", "gamma2_1"), "phi2": ("mu2", "gamma2_2")}
    fixed_effets = {}
    obs = {"time": jnp.linspace(60, 95, num=J_OBS)}

    obs2, sim, PRNGKey = mem_simulation(
        params,
        PRNGKey,
        N_IND,
        "sigma2",
        linear_fct,
        random_effects,
        fixed_effets,
        fct_kwargs={"time": obs["time"]},
    )

    obs.update(obs2)

    return obs, sim, PRNGKey


def cox_weibull_simulation(params, PRNGKey, N_IND, logistic_sim):
    baseline_kwargs = {"a": params.a, "b": params.b}

    def baseline_fct(t, a, b):
        return b / a * (t / a) ** (b - 1)

    link_kwargs = {
        "phi1": logistic_sim["phi1"],
        "phi2": logistic_sim["phi2"],
    }

    def link_fct(t, alpha, phi1, phi2):
        return alpha * linear_curve_float(t, phi1, phi2)

    DIM_COV = params.beta.shape[0]
    cov, PRNGKey = cov_simulation(PRNGKey, min=-1, max=1, shape=(N_IND, DIM_COV))

    _, sim, PRNGKey = cox_simulation(
        params,
        PRNGKey,
        cov @ params.beta,
        baseline_fct,
        baseline_kwargs,
        link_fct,
        link_kwargs,
    )

    return {"cov": cov}, sim, PRNGKey


def sample(
    params0_weibull,
    PRNGKey,
    N_IND,
    J_OBS,
    weibull_censoring_loc,
):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    obs, sim, PRNGKey = nlmem_simulation(
        params0_weibull, PRNGKey, N_IND, J_OBS, t_min=60, t_max=135
    )

    obs2, sim2, PRNGKey = cox_weibull_simulation(
        params0_weibull, PRNGKey, N_IND, logistic_sim=sim
    )

    obs.update(obs2)
    sim.update(sim2)

    rng = np.random.default_rng()
    C = weibull_censoring_loc * rng.weibull(35, len(sim["T uncensored"]))
    C = np.minimum(C, obs["time"].max())

    T = np.minimum(sim["T uncensored"], C)
    delta = sim["T uncensored"] < C

    obs.update({"T": T, "delta": delta})
    sim["C"] = C

    # obs["Y"] = obs["Y"][:, obs["time"] < obs["T"].max()]
    # obs["time"] = obs["time"][obs["time"] < obs["T"].max()]

    return obs, sim, PRNGKey


def get_solver(PRNGKey, params0, data_set, likelihood, likelihood_array):
    key1, key_out = jrd.split(PRNGKey, num=2)
    N_IND, _ = data_set["Y"].shape
    (DIM_COV,) = params0["beta"].shape

    parametrization, _, PRNGKey = get_parametrization(PRNGKey, DIM_COV)
    # ====== SOLVER CREATION ====== #
    solver = Gradient(key1, parametrization, params0)

    # ============================================================ #
    # ==================== MCMC configuration ==================== #
    # ============================================================ #
    if isinstance(params0, dict):
        mu1 = params0["mu1"]
        mu2 = params0["mu2"]
    else:
        mu1 = params0.mu1
        mu2 = params0.mu2

    solver.add_mcmc(
        float(mu1),
        sd=0.001,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi1",
    )
    solver.latent_variables["phi1"].adaptative_sd = True
    solver.add_mcmc(
        float(mu2),
        sd=0.0001,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi2",
    )
    solver.latent_variables["phi2"].adaptative_sd = True
    # ============================================================ #
    # ==================== END configuration ==================== #
    # ============================================================ #

    solver.add_data(**data_set)
    solver.likelihood = likelihood
    solver.add_likelihood_kwargs("time", "Y", "phi1", "phi2", "T", "delta", "cov")

    solver.add_data(parametrization=solver.parametrization)
    solver.add_likelihood_kwargs("parametrization")

    return solver, key_out


def get_parametrization(PRNGKey, DIM_COV):
    PRNGKey, key1, key2, key3 = jrd.split(PRNGKey, num=4)

    parametrization = pc.NamedTuple(
        mu1=pc.RealPositive(scale=0.05),
        mu2=pc.Real(scale=0.01),
        gamma2_1=pc.RealPositive(scale=0.01),
        gamma2_2=pc.RealPositive(scale=0.000001),
        sigma2=pc.RealPositive(scale=0.0001),
        alpha=pc.Real(scale=0.1),
        beta=pc.Real(scale=1, shape=(DIM_COV,)),
    )

    params_star_weibull = params_weibull(
        mu1=0.2,
        mu2=0.01,
        gamma2_1=0.01,
        gamma2_2=0.000005,
        sigma2=1e-8,
        a=85.0,
        b=35,
        alpha=0.95,
        beta=jnp.concatenate(
            [jnp.array([-2, -3, 3, 2]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    return parametrization, params_star_weibull, PRNGKey


def get_params_star(PRNGKey, DIM_COV):
    parametrization, params_star_weibull, PRNGKey = get_parametrization(
        PRNGKey, DIM_COV
    )

    params_star_stack = np.hstack(
        [
            params_star_weibull.mu1,
            params_star_weibull.mu2,
            params_star_weibull.gamma2_1,
            params_star_weibull.gamma2_2,
            params_star_weibull.sigma2,
            params_star_weibull.alpha,
            params_star_weibull.beta,
        ]
    )

    return params_star_stack, params_star_weibull, PRNGKey


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    N_IND = 500
    J_OBS = 10

    _, params_star, PRNGKey = get_parametrization(jrd.PRNGKey(0), 10)

    censoring_loc = 85
    obs, sim, PRNGKey = sample(
        params_star,
        PRNGKey=PRNGKey,
        N_IND=N_IND,
        J_OBS=J_OBS,
        weibull_censoring_loc=censoring_loc,
    )

    plt.plot(obs["time"].T, obs["Y"].T, "o-")

    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(7)
    plt.hist(
        [obs["T"], sim["T uncensored"]],  # , sim["C"]],
        bins=20,
        density=True,
        label=["censored survival time", "survival time"],  # , "censuring time"],
    )
    baseline_fct = (
        lambda t: params_star.b
        / params_star.a
        * (t / params_star.a) ** (params_star.b - 1)
        * np.exp(-((t / params_star.a) ** params_star.b))
    )

    t = np.linspace(obs["T"].min(), 105, num=100)
    plt.plot(t, baseline_fct(t), label="weibull baseline")
    weibull_fct = (
        lambda t: 35
        / censoring_loc
        * (t / censoring_loc) ** (35 - 1)
        * np.exp(-((t / censoring_loc) ** 35))
    )

    plt.plot(t, weibull_fct(t), label="censured time weibull distribution")
    plt.legend()

    print(f'censoring = {int((1-obs["delta"].mean())*100)}%')
