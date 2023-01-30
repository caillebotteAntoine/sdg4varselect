import numpy as np

np.set_printoptions(precision=3, threshold=10)


def default_arg(dec):
    def new_dec(func=None, *args, **kwargs):
        if func is None:

            def inner_dec(func):
                return dec(func, *args, **kwargs)

            return inner_dec

        return dec(func, *args, **kwargs)

    return new_dec


@default_arg
def time_profiler(func=None, nrun=10, running_time_max=60):
    import time

    def new_func(*args, **kwargs):
        start = time.time()

        i = 0
        elasped_time = 0
        while i < nrun and elasped_time < running_time_max:
            out = func(*args, **kwargs)
            i += 1
            elasped_time = time.time() - start

        msg = "{:8.3f}".format(elasped_time / i)
        print(str(i) + " loops: " + msg + " s per loop")
        return out

    return new_func


def loadbar(os: str, progress: float, maxbar: int = 50, indicator: str = ">") -> str:
    nbar = int(progress * maxbar)
    if nbar > maxbar:
        nbar = maxbar

    os += "["
    for i in range(nbar):
        os += "="
    os += indicator
    for i in range(nbar, maxbar):
        os += " "
    os += "]"
    return os


def loadnumber(os: str, number: int, max: int, unit: str = "") -> str:
    number_str = str(number)
    max_str = str(max)

    if len(number_str) < len(max_str):
        diff_len = len(max_str) - len(number_str)
        for i in range(diff_len):
            os += " "

    os += number_str + "/" + max_str + unit
    return os


def step_message(iter: int, max_iter: int) -> None:
    os = ""
    os = loadnumber(os, iter, max_iter) + " "
    os = loadbar(os, float(iter) / max_iter)
    print(os, end="\r")


if __name__ == "__main__":

    import numpy as np

    x = np.arange(0, 1e6)

    @time_profiler
    def np_sum_test(x):
        np.sum(x)

    @time_profiler(nrun=5)
    def sum_test(x):
        sum(x)

    np_sum_test(x)
    sum_test(x)
