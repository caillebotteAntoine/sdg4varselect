from datetime import datetime
import pickle
from sdg4varselect.miscellaneous import step_message

# joint_model
from joint_model.sample import get_params_star, sample
from joint_model.one_run import get_random_params0
from joint_model.one_res import method

from time import time
from sdg4varselect import jrd, jnp


# ====================================================== #
def multi_run(
    params0,
    lbd_set,
    N_IND,
    DIM_COV,
    J_OBS,
    CENSORING,
    nrun,
    seed=None,
):
    if not isinstance(CENSORING, list):
        CENSORING = [CENSORING]

    for censoring in CENSORING:
        msg = f"n = {N_IND}, p = {DIM_COV}, J = {J_OBS}, C = {censoring}"
        msg = "\n" + "=" * len(msg) + "\n" + msg + "\n" + "=" * len(msg)
        print(msg)

        if seed is None:
            seed = int(time())  #
        print(f"seed = {seed}")
        prng_key = jrd.PRNGKey(seed)

        print(f'start at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        # ====================================================== #
        params_star_stack, params_star_weibull, prng_key = get_params_star(
            prng_key, DIM_COV
        )
        # ====================================================== #

        lrf = []
        lrs = []
        llbd = []
        lbic = []
        lebic = []
        ltheta_reg = []
        lcensoring_rate = []

        for i in range(nrun):
            print("\nrun = " + step_message(i, nrun), end="\n")

            print(f"prng_key = {prng_key}")

            # ================ DATA SET GENERATION ================= #
            data_set, _, prng_key = sample(
                params_star_weibull, prng_key, N_IND, J_OBS, censoring
            )
            censoring_rate = 1 - data_set["delta"].mean()
            print(f"censoring = {int(censoring_rate*100)}%")

            # ================ ESTIMATION ================= #
            params_start, prng_key = get_random_params0(prng_key, params0, error=0.2)

            res = method(
                params_start,
                data_set,
                lbd_set,
                N_IND,
                DIM_COV,
                prng_key=prng_key,
                verbatim=False,
            )
            if res != -1:
                res_f, _, solver, res_s, bic, ebic, theta_reg, lbd_select, _, _ = res
                lrf.append(res_f)
                lrs.append(res_s)
                llbd.append(lbd_select)
                lbic.append(bic)
                lebic.append(ebic)
                ltheta_reg.append(theta_reg)
                lcensoring_rate.append(censoring_rate)

        print("\nrun = " + step_message(nrun, nrun), end="\n")

        print(f'end at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

        theta = [res.theta[-1] for res in lrf]
        theta_biais = [res.theta[-1] for res in lrs]

        data = {
            "theta": theta,
            "theta_biais": theta_biais,
            "lbd_select": llbd,
            # "params_star_weibull": params_star_weibull,
            "params_star_stack": params_star_stack,
            "params_names": solver.params_names,
            "lbic": lbic,
            "lebic": lebic,
            "ltheta_reg": ltheta_reg,
            "lbd_set": lbd_set,
            "lcensoring_rate": lcensoring_rate,
            "seed": seed,
        }

        filename = f"{int(seed)}_multi_N{N_IND}_P{DIM_COV}"

        filename += f"_J{J_OBS}_C{int(jnp.array(lcensoring_rate).mean()*100)}"

        with open(f"results/{filename}.pkl", "wb") as f:
            pickle.dump(data, f)

        print("RESULT SAVED !\n\n")
    return True
