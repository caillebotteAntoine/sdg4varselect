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
params_type = namedtuple(
    "params_type",
    (
        "mu1",
        "mu2",
        "gamma2_1",
        "gamma2_2",
        "sigma2",
        "alpha",
        "beta_surv",
        "beta_long",
    ),
)


# ======================================================= #
# ====================== SIMULATION ===================== #
# ======================================================= #
def lmem_simulation(params, PRNGKey, N_IND, J_OBS):
    def linear_fct(time, phi1, phi2, beta_prod_cov):
        return linear_curve(
            time=time,
            intercept=phi1 + beta_prod_cov,
            slope=phi2,
        )

    random_effects = {"phi1": ("mu1", "gamma2_1"), "phi2": ("mu2", "gamma2_2")}
    fixed_effets = {}
    obs = {}

    DIM_COV_LONG = params.beta_long.shape[0]
    obs["cov_long"], PRNGKey = cov_simulation(
        PRNGKey, min=-0.1, max=0.1, shape=(N_IND, DIM_COV_LONG)
    )

    # mimicking yearly appointments
    time = jnp.arange(J_OBS) * 365
    time = jnp.concatenate(jnp.array([[time]] * N_IND), axis=0)
    time += jrd.choice(jrd.PRNGKey(0), jnp.arange(1, 365), time.shape)
    time /= time.shape[1] * 365
    obs["time"] = time

    obs2, sim, PRNGKey = mem_simulation(
        params,
        PRNGKey,
        N_IND,
        "sigma2",
        linear_fct,
        random_effects,
        fixed_effets,
        fct_kwargs={
            "time": obs["time"],
            "beta_prod_cov": obs["cov_long"] @ params.beta_long,
        },
    )

    obs.update(obs2)

    return obs, sim, PRNGKey


def cox_censured_simulation(
    params, PRNGKey, N_IND, linear_sim, beta_prod_cov_long, censoring_time
):
    baseline_kwargs = {"a": 10, "b": 1.1}

    def baseline_fct(t, a, b):
        return b / a * (t / a) ** (b - 1)

    link_kwargs = {
        "phi1": linear_sim["phi1"],
        "phi2": linear_sim["phi2"],
        "beta_prod_cov_long": beta_prod_cov_long,
    }

    def link_fct(t, alpha, phi1, phi2, beta_prod_cov_long):
        return alpha * linear_curve_float(
            t, intercept=phi1 + beta_prod_cov_long, slope=phi2
        )

    DIM_COV = params.beta_surv.shape[0]
    cov_surv, PRNGKey = cov_simulation(
        PRNGKey, min=-0.1, max=0.1, shape=(N_IND, DIM_COV)
    )

    obs, sim, PRNGKey = cox_simulation(
        params,
        PRNGKey,
        cov_surv @ params.beta_surv,
        baseline_fct,
        baseline_kwargs,
        link_fct,
        link_kwargs,
    )

    delta = (
        jnp.ones(sim["T uncensored"].shape)
        if censoring_time is None
        else sim["T uncensored"] <= censoring_time
    )
    T = (
        sim["T uncensored"]
        if censoring_time is None
        else jnp.minimum(sim["T uncensored"], censoring_time)
    )

    obs = {"T": T, "delta": delta, "cov_surv": cov_surv}

    return obs, sim, PRNGKey


def sample(params0, PRNGKey, N_IND, J_OBS, *args, **kwargs):
    """return longitudinal and survival simulation
    and latente variable simulation in the two dict"""
    obs, sim, PRNGKey = lmem_simulation(params0, PRNGKey, N_IND, J_OBS)

    DIM_COV_SURV = params0.beta_surv.shape[0]

    cov_surv = jrd.uniform(
        PRNGKey, minval=-0.1, maxval=0.1, shape=(N_IND, DIM_COV_SURV)
    )
    cov_surv = cov_surv - cov_surv.mean(axis=0)[None, :]
    cov_surv = jnp.array(
        cov_surv,
        dtype=jnp.float32,
    )

    obs2, sim2, PRNGKey = cox_censured_simulation(
        params0,
        PRNGKey,
        N_IND,
        sim,
        obs["cov_long"] @ params0.beta_long,
        censoring_time=obs["time"].max(),
    )

    obs.update(obs2)
    sim.update(sim2)

    return obs, sim, PRNGKey


def get_solver(PRNGKey, params0, data_set, likelihood, likelihood_array):
    key1, key_out = jrd.split(PRNGKey, num=2)
    N_IND, _ = data_set["Y"].shape
    (DIM_COV,) = params0["beta_long"].shape

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
        sd=0.5,
        size=N_IND,
        likelihood=likelihood_array,
        name="phi1",
    )
    solver.latent_variables["phi1"].adaptative_sd = True
    solver.add_mcmc(
        float(mu2),
        sd=0.5,
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
    solver.add_likelihood_kwargs(
        "time", "Y", "phi1", "phi2", "T", "delta", "cov_surv", "cov_long"
    )

    solver.add_data(parametrization=solver.parametrization)
    solver.add_likelihood_kwargs("parametrization")

    return solver, key_out


def get_parametrization(PRNGKey, DIM_COV):
    PRNGKey, key1, key2, key3 = jrd.split(PRNGKey, num=4)

    parametrization = pc.NamedTuple(
        mu1=pc.RealPositive(scale=0.1),
        mu2=pc.Real(scale=0.1),
        gamma2_1=pc.RealPositive(scale=1),
        gamma2_2=pc.RealPositive(scale=1),
        sigma2=pc.RealPositive(scale=0.01),
        alpha=pc.Real(scale=0.1),
        beta_surv=pc.Real(scale=1, shape=(DIM_COV,)),
        beta_long=pc.Real(scale=1, shape=(DIM_COV,)),
    )

    params_star = params_type(
        mu1=1,
        mu2=1.5,
        gamma2_1=2**2,
        gamma2_2=0.3**2,
        sigma2=0.1**2,
        alpha=0.0001,
        beta_surv=jnp.concatenate(
            [jnp.array([10, 0.5, 0.3, 0.5]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
        beta_long=jnp.concatenate(
            [jnp.array([1, 2, 1, 2]), jnp.zeros(shape=(DIM_COV - 4,))]
        ),
    )

    return parametrization, params_star, PRNGKey


def get_params_star(PRNGKey, DIM_COV):
    parametrization, params_star, PRNGKey = get_parametrization(PRNGKey, DIM_COV)

    params_star_stack = np.hstack(
        [
            params_star.mu1,
            params_star.mu2,
            params_star.gamma2_1,
            params_star.gamma2_2,
            params_star.sigma2,
            params_star.alpha,
            params_star.beta_surv,
            params_star.beta_long,
        ]
    )

    return params_star_stack, params_star, PRNGKey


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    DIM_COV = 10
    _, params_star, PRNGKey = get_parametrization(jrd.PRNGKey(0), DIM_COV)

    N_IND = 500
    J_OBS = 5

    obs, _, PRNGKey = sample(
        params_star,
        PRNGKey=PRNGKey,
        N_IND=N_IND,
        J_OBS=J_OBS,
    )

    plt.plot(obs["time"].T, obs["Y"].T, "o-")

    plt.figure()
    plt.hist(obs["T"][obs["delta"]], bins=20)

    print(f'censoring = {int((1-obs["delta"].mean())*100)}%')

if __name__ == "zut":
    # =============================================== #
    # ========= COMPARAISON FCT REPARTITION ========= #
    # =============================================== #
    obs, sim, PRNGKey = lmem_simulation(params_star, jrd.PRNGKey(0), N_IND, J_OBS)

    cov_surv = jrd.uniform(PRNGKey, minval=-0.1, maxval=0.1, shape=(N_IND, DIM_COV))
    cov_surv = cov_surv - cov_surv.mean(axis=0)[None, :]
    cov_surv = jnp.array(
        cov_surv,
        dtype=jnp.float32,
    )
    # Test
    cov_surv = np.repeat(cov_surv[1][:, None], N_IND, axis=1).T
    cov_long = np.repeat(obs["cov_long"][1][:, None], N_IND, axis=1).T
    sim["phi1"] = np.repeat(sim["phi1"][1], N_IND)
    sim["phi2"] = np.repeat(sim["phi2"][1], N_IND)

    obs2, sim2, PRNGKey = cox_censured_simulation(
        params_star,
        PRNGKey,
        N_IND,
        sim,
        cov_long @ params_star.beta_long,
        censoring_time=None,  # obs["time"].max()
    )

    obs.update(obs2)
    sim.update(sim2)

    def fct_rep(t, beta_prod_cov_long, beta_prod_cov_surv, phi1, phi2):
        """
        return P(T <= t ) = 1 - S(t)

        where S is the survival function  : S(t) = exp(-int_0^t lbd(s) ds )
        where lbd is the hazard function : lbd(t) = lbd0(t) * exp(beta^T U  + alpha* m(t))
                                    with : lbd0(t) = b a^-b t^{b-1} = b /a * (t/a)^{b-1}
        """
        if isinstance(t, (np.ndarray, jnp.ndarray)) and t.shape != ():
            return np.array(
                [
                    fct_rep(ti, beta_prod_cov_long, beta_prod_cov_surv, phi1, phi2)
                    for ti in t
                ]
            )

        def lbd(u):
            return jnp.exp(
                beta_prod_cov_surv
                + params_star.alpha
                * linear_curve_float(u, intercept=phi1 + beta_prod_cov_long, slope=phi2)
            )

        t_linspace = jnp.linspace(0, t, num=100)
        return 1 - jnp.exp(-jnp.trapz(y=lbd(t_linspace), x=t_linspace))

    def fct_rep_empirique(t, T):
        if isinstance(t, (np.ndarray, jnp.ndarray)) and t.shape != ():
            return np.array([fct_rep_empirique(ti, T) for ti in t])
        return (T < t).mean()

    beta_prod_cov_long = (cov_long @ params_star.beta_long)[1]
    beta_prod_cov_surv = (cov_surv @ params_star.beta_surv)[1]
    phi1 = sim["phi1"][1]
    phi2 = sim["phi2"][1]

    t_linspace = jnp.linspace(obs["T"].min(), obs["T"].max(), num=100)

    plt.figure()
    _ = plt.hist(obs["T"], bins=20)

    plt.figure()
    plt.plot(t_linspace, fct_rep_empirique(t_linspace, obs["T"]), label="empirique")
    plt.plot(
        t_linspace,
        fct_rep(t_linspace, beta_prod_cov_long, beta_prod_cov_surv, phi1, phi2),
        label="théorique",
    )
    plt.legend()
