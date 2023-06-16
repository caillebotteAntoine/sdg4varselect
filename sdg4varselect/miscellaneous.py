import numpy as np
import jax.numpy as jnp

# from warnings import warn

np.set_printoptions(precision=3, threshold=10)


def default_arg(dec):
    def new_dec(func=None, *args, **kwargs):
        if func is None:

            def inner_dec(func):
                return dec(func, *args, **kwargs)

            return inner_dec

        return dec(func, *args, **kwargs)

    return new_dec


def time2string(x: float, format="{:4.3f}"):
    unit = np.array([1e-1, 1e-4, 1e-7, 1e-10])
    r = x >= unit
    unit_txt = ["s", "ms", "µs", "ns"]
    i = r.argmax()
    out = format.format(x / unit[i] / 10) + unit_txt[i]
    return out


def difftime2saving(nloop, difftime, unit="s"):
    if difftime < 0:
        return difftime2saving(nloop, -difftime, unit)

    units = {"s": 1, "ms": 1e-3, "µs": 1e-6, "ns": 1e-9}

    msg = (
        "An acceleration of "
        + time2string(difftime * units[unit])
        + " in a "
        + str(nloop)
        + " iteration loops will allow a gain of "
        + time2string(difftime * nloop * units[unit])
    )
    return msg


@default_arg
def time_profiler(func=None, nrun=10, running_time_max=60):
    from jaxlib.xla_extension import CompiledFunction
    import time

    def new_func(*args, **kwargs):
        # ensure that the function has been compiled before measuring the computation time
        if isinstance(func, CompiledFunction):
            # warn("the function was compiled before timing")
            func(*args, **kwargs)

        elasped_time = [time.time()]
        while (
            len(elasped_time) < nrun + 1
            and time.time() - elasped_time[0] < running_time_max
        ):
            out = func(*args, **kwargs)
            elasped_time.append(time.time())

        elasped_time = np.array(elasped_time)
        elasped_time = elasped_time[1:] - elasped_time[:-1]

        msg = time2string(elasped_time.mean()) + " per loop"

        if elasped_time.var() > 1e-3:
            msg += " [sd = " + time2string(np.sqrt(elasped_time.var())) + "]"
        print(str(len(elasped_time)) + " loops: " + msg)
        return out, elasped_time.mean()

    return new_func


def difftime(*funcs, nloop=100, nrun=20):
    def func(*args, **kwargs):
        func_time = np.array([0.0 for i in range(len(funcs))])
        for i in range(len(funcs)):
            print("[ " + funcs[i].__name__ + " ] ", end="")
            out, tmp = time_profiler(funcs[i], nrun=nrun)(*args, **kwargs)
            func_time[i] = tmp

        loop_times_msg = (
            " out of a total of " + time2string(nloop * func_time.min()) + ".\n"
        )
        difftime2saving(nloop, func_time.max() - func_time.min(), end=loop_times_msg)
        print("The fastest function is " + funcs[func_time.argmin()].__name__)

    return func


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


def step_message(iter: int, max_iter: int) -> str:
    os = ""
    os = loadnumber(os, iter, max_iter) + " "
    os = loadbar(os, float(iter) / max_iter)
    return os  # you should use `print(os, end="\r")`


if __name__ == "__main__":

    import numpy as np

    x = np.arange(0, 1e6)

    def np_sum(x):
        np.sum(x)

    def mysum(x):
        res = 0
        for v in x:
            res += v
        return res

    difftime(np_sum, sum, mysum)(x)

    print([time2string(x) for x in [2.5, 0.6, 2.3e-4, 2.3e-3, 2.2e-7, 2.2e-6, 2.1e-10]])
