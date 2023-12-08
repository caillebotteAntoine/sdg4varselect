import numpy as np
import parametrization_cookbook.jax as pc
from sdg4varselect.data_generation import mem_simulation, cox_simulation, cov_simulation

from sdg4varselect import Gradient, jrd, jnp
from collections import namedtuple


from sdg4varselect.logistic_model import (
    logistic_curve,
    logistic_curve_float,
)

# ============================================================= #
# ====================== PARAMIETRIZATION ===================== #
# ============================================================= #
params_weibull = namedtuple(
    "params_weibull",
    ("mu1", "mu2", "mu3", "gamma2_1", "gamma2_2", "sigma2", "a", "b", "alpha", "beta"),
)


# ===================================================== #
# ================== DATA GENERATION ================== #
# ===================================================== #
def nlmem_simulation(params, PRNGKey, N_IND, J_OBS, t_min, t_max):
    """return longitudinal simulation
    and latente variable simulation in the two dict"""

    def logistic_fct(time, phi1, phi2, phi3):
        return logistic_curve(
            time=time,
            supremum=phi1,
            midpoint=phi2,
            growth_rate=phi3,
        )

    random_effects = {"phi1": ("mu1", "gamma2_1"), "phi2": ("mu2", "gamma2_2")}
    fixed_effets = {"phi3": "mu3"}

    time = jnp.linspace(60, 135, num=J_OBS)
    # time = jnp.concatenate(jnp.array([[time]] * N_IND), axis=0)
    # time += jrd.uniform(jrd.PRNGKey(0), minval=-2, maxval=2, shape=time.shape)

    obs = {"time": time}

    obs2, sim, PRNGKey = mem_simulation(
        params,
        PRNGKey,
        N_IND,
        "sigma2",
        logistic_fct,
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
        "mu3": params.mu3,
    }

    def link_fct(t, alpha, phi1, phi2, mu3):
        return alpha * logistic_curve_float(t, phi1, phi2, mu3)

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
    prng_key,
    N_IND,
    J_OBS,
    CENSORING,
):
    from test import data_simulation

    """return data_set, sim, prng_key"""
    # ====== DATA GENERATION ====== #
    data_set, sim, prng_key = data_simulation(
        params=params0_weibull,
        key=prng_key,
        N_IND=N_IND,
        J_OBS=J_OBS,
        t_min=60,
        t_max=135,
        censoring=CENSORING,
    )
    return data_set, sim, prng_key


def sample2(
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
    # C = np.minimum(C, obs["time"].max() + 1)

    T = np.minimum(sim["T uncensored"], C)
    delta = sim["T uncensored"] < C

    obs.update({"T": T, "delta": delta})
    sim["C"] = C

    # obs["Y"] = np.array(
    #     [np.where(obs["time"][i] < T[i], obs["Y"][i], np.nan) for i in range(len(T))]
    # )
    # obs["time"] = np.array(
    #     [np.where(obs["time"][i] < T[i], obs["time"][i], np.nan) for i in range(len(T))]
    # )

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
        sd=2,
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
        mu1=pc.RealPositive(scale=0.5),
        mu2=pc.Real(scale=100),
        mu3=pc.RealPositive(scale=5),
        gamma2_1=pc.RealPositive(scale=0.001),
        gamma2_2=pc.RealPositive(scale=10),
        # a=pc.Real(scale=100),
        # b=pc.Real(scale=50),
        sigma2=pc.RealPositive(scale=0.001),
        alpha=pc.Real(scale=10),
        beta=pc.Real(scale=1, shape=(DIM_COV,)),
    )

    params_star_weibull = params_weibull(
        mu1=0.3,
        mu2=90.0,
        mu3=7.5,
        gamma2_1=0.0025,
        gamma2_2=20,
        sigma2=0.001,
        a=130.0,
        b=35,
        alpha=11.11,
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
            params_star_weibull.mu3,
            params_star_weibull.gamma2_1,
            params_star_weibull.gamma2_2,
            # params_star_weibull.a,
            # params_star_weibull.b,
            params_star_weibull.sigma2,
            params_star_weibull.alpha,
            params_star_weibull.beta,
        ]
    )

    return params_star_stack, params_star_weibull, PRNGKey


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from time import time

    N_IND = 10
    J_OBS = 5

    _, params_star, PRNGKey = get_parametrization(jrd.PRNGKey(int(time())), 10)

    def test_censoring_loc(censoring_loc, PRNGKey):
        obs, sim, PRNGKey = sample(
            params_star,
            PRNGKey=PRNGKey,
            N_IND=N_IND,
            J_OBS=J_OBS,
            weibull_censoring_loc=censoring_loc,
        )

        fig = plt.figure()
        fig.set_figheight(7)
        fig.set_figwidth(7)

        ax = fig.add_subplot(211)
        ax.plot(obs["time"].T, obs["Y"].T, "o-")

        ax = fig.add_subplot(212)
        ax.hist(
            [obs["T"], sim["T uncensored"]],  # , sim["C"]],
            bins=20,
            density=True,
            label=["censored survival time", "survival time"],  # , "censuring time"],
        )

        def weibull_fct(t, a, b):
            return b / a * (t / a) ** (b - 1) * np.exp(-((t / a) ** b))

        t = np.linspace(
            obs["T"].min(), max(obs["T"].max(), sim["T uncensored"].max()), num=100
        )
        ax.plot(
            t, weibull_fct(t, params_star.a, params_star.b), label="weibull baseline"
        )

        ax.plot(
            t,
            weibull_fct(t, censoring_loc, 35),
            label="censured time weibull distribution",
        )
        ax.legend()

        print(f'censoring = {int((1-obs["delta"].mean())*100)}%')

        fig.suptitle(
            f'Simulation with {int((1-obs["delta"].mean())*100)}% censored data'
        )
        return obs, sim, PRNGKey

    obs, sim, PRNGKey = test_censoring_loc(1000, PRNGKey)
    # obs, sim, PRNGKey = test_censoring_loc(128, PRNGKey)  # ~20%
    # obs, sim, PRNGKey = test_censoring_loc(122, PRNGKey)  # ~40%
    # obs, sim, PRNGKey = test_censoring_loc(116, PRNGKey)  # ~60%
    # obs, sim, PRNGKey = test_censoring_loc(110, PRNGKey)  # ~80%


if __name__ == " zut":
    # =============================================== #
    # ========= COMPARAISON FCT REPARTITION ========= #
    # =============================================== #
    obs, sim, PRNGKey = nlmem_simulation(
        params_star, jrd.PRNGKey(0), N_IND, J_OBS, t_min=60, t_max=135
    )

    baseline_kwargs = {"a": params_star.a, "b": params_star.b}

    def baseline_fct(t, a, b):
        return b / a * (t / a) ** (b - 1)

    link_kwargs = {
        "phi1": sim["phi1"],
        "phi2": sim["phi2"],
        "mu3": params_star.mu3,
    }

    def link_fct(t, alpha, phi1, phi2, mu3):
        return alpha * logistic_curve_float(t, phi1, phi2, mu3)

    DIM_COV = params_star.beta.shape[0]
    cov, PRNGKey = cov_simulation(PRNGKey, min=-1, max=1, shape=(N_IND, DIM_COV))

    # Test
    cov = np.repeat(cov[1][:, None], N_IND, axis=1).T
    sim["phi1"] = np.repeat(sim["phi1"][1], N_IND)
    sim["phi2"] = np.repeat(sim["phi2"][1], N_IND)

    obs2, sim2, PRNGKey = cox_simulation(
        params_star,
        PRNGKey,
        cov @ params_star.beta,
        baseline_fct,
        baseline_kwargs,
        link_fct,
        link_kwargs,
    )

    obs.update(obs2)
    sim.update(sim2)

    def fct_rep(t, beta_prod_cov, phi1, phi2):
        """
        return P(T <= t ) = 1 - S(t)

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}
        """
        if isinstance(t, (np.ndarray, jnp.ndarray)) and t.shape != ():
            return np.array([fct_rep(ti, beta_prod_cov, phi1, phi2) for ti in t])

        def lbd(u):
            lbd0 = (
                params_star.b
                / params_star.a
                * (u / params_star.a) ** (params_star.b - 1)
            )
            return lbd0 * jnp.exp(
                beta_prod_cov
                + params_star.alpha
                * logistic_curve_float(u, phi1, phi2, params_star.mu3)
            )

        t_linspace = jnp.linspace(0, t, num=100)
        return 1 - jnp.exp(-jnp.trapz(y=lbd(t_linspace), x=t_linspace))

    def fct_rep_empirique(t, T):
        if isinstance(t, (np.ndarray, jnp.ndarray)) and t.shape != ():
            return np.array([fct_rep_empirique(ti, T) for ti in t])
        return (T < t).mean()

    beta_prod_cov = (cov @ params_star.beta)[1]
    phi1 = sim["phi1"][1]
    phi2 = sim["phi2"][1]

    t_linspace = jnp.linspace(
        sim["T uncensored"].min(), sim["T uncensored"].max(), num=100
    )

    plt.figure()
    _ = plt.hist(sim["T uncensored"], bins=20)

    plt.figure()
    plt.plot(
        t_linspace,
        fct_rep_empirique(t_linspace, sim["T uncensored"]),
        label="empirique",
    )
    plt.plot(
        t_linspace, fct_rep(t_linspace, beta_prod_cov, phi1, phi2), label="théorique"
    )
    plt.legend()
