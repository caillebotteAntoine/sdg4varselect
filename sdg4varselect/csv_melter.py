import csv

from sdg4varselect.chain import chain
from sdg4varselect.MCMC import MCMC_chain
from sdg4varselect.parameter import parameter


def csv_melter(writer, x) -> None:
    if isinstance(x, (chain, parameter, MCMC_chain)):
        c = x.chain()
        for iteration in range(len(c)):
            for id in range(len(c[iteration])):
                writer.writerow([id, c[iteration][id], iteration, x.type(), x.name()])

        if isinstance(x, MCMC_chain):
            a = x.acceptance_rate()
            for iteration in range(len(a)):
                writer.writerow([0, a[iteration], iteration, "acceptance", x.name()])

            sd = x.sd()
            for iteration in range(len(sd)):
                writer.writerow([0, sd[iteration], iteration, "sd", x.name()])
    else:
        raise TypeError("x must be a solver, chain, a parameter or a MCMC_chain")


def solver2csv(
    elapsed_time, parameters, latent_variables, step_size, iter, file_name: str
) -> None:
    file = open(file_name, "w", newline="")
    writer = csv.writer(file, delimiter=";")
    writer.writerow(["id", "value", "iteration", "var_type", "var_name"])
    writer.writerow([0, elapsed_time, 0, "elapsed_time", "elapsed_time"])

    for par in parameters.values():
        csv_melter(writer, par)

    for var in latent_variables.values():
        csv_melter(writer, var)

    for i in range(iter):
        writer.writerow([0, step_size(i), i, "learning_rate", "learning_rate"])

    file.close()


if __name__ == "__main__":
    x = chain(-1, 5)
    for i in range(10):
        for j in range(len(x)):
            x.data()[j] = i

        x.update_chain()

    print(x)
    print(x.chain())

    file = open("test", "w", newline="")
    writer = csv.writer(file, delimiter=";")
    writer.writerow(["id", "value", "iteration", "var_type", "var_name"])

    csv_melter(writer, x)
