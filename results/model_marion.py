

    import matplotlib.pyplot as plt
    from sdg4varselect.models.hd_logistic_mem import (
        HDLogisticMixedEffectsModel,
    )
    import jax.random as jrd

    myModel = HDLogisticMixedEffectsModel(N=100, J=10, P=5)

    my_params_star = myModel.new_params(
        eta1=200,
        eta2=300,
        gamma2_1=0.1,
        gamma2_2=0.1,
        mu=1200,
        sigma2=30,
        Gamma2=200,
        beta=jnp.concatenate(
            [jnp.array([100, 50, 20]), jnp.zeros(shape=(myModel.P - 3,))]
        ),
    )

    obs, sim = myModel.sample(my_params_star, jrd.PRNGKey(0))

    plt.plot(obs["mem_obs_time"].T, obs["Y"].T, "o-")

    algo_settings = GradientDescentFIM.GradientDescentFIMSettings(
        step_size_grad={
            "learning_rate": 1e-8,
            "preheating": 400,
            "heating": 600,
            "max": 0.9,
        },
        step_size_approx_sto={
            "learning_rate": 1e-8,
            "preheating": 400,
            "heating": None,
            "max": 1,
        },
        step_size_fisher={
            "learning_rate": 1e-8,
            "preheating": 400,
            "heating": None,
            "max": 0.9,
        },
    )

    algo = StochasticGradientDescentFIM(jrd.PRNGKey(0), 1000, algo_settings)
    # =================== MCMC configuration ==================== #
    algo.add_mcmc(
        float(my_params_star.eta1),
        sd=0.01,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="psi1",
    )
    algo.latent_variables["psi1"].adaptative_sd = True
    algo.add_mcmc(
        float(my_params_star.eta2),
        sd=0.01,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="psi2",
    )
    algo.latent_variables["psi2"].adaptative_sd = True
    algo.add_mcmc(
        float(my_params_star.eta2),
        sd=10,
        size=myModel.N,
        likelihood=myModel.likelihood_array,
        name="ksi",
    )
    algo.latent_variables["ksi"].adaptative_sd = True
    # ==================== END configuration ==================== #

    res = []
    for i in range(10):
        theta0 = 0.2 * jrd.normal(jrd.PRNGKey(i), shape=(myModel.parametrization.size,))
        res.append(algo.fit(myModel, obs, theta0, partial_fit=True))

    import sdg4varselect.plot as sdgplt

    sdgplt.plot_theta(res, 7, my_params_star, myModel.params_names)
    sdgplt.plot_theta_HD(res, 7, my_params_star, myModel.params_names)
