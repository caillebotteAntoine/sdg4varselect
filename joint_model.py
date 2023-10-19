import numpy as np

np.__config__.show()

import os

# print(os.environ["OMP_NUM_THREADS"])
# print(os.environ["OPENBLAS_NUM_THREADS"])
# print(os.environ["MKL_NUM_THREADS"])
# print(os.environ["VECLIB_MAXIMUM_THREADS"])
# print(os.environ["NUMEXPR_NUM_THREADS"])

import time, os, jax, numpy as np, jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")  # insures we use the CPU

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


def timer(name, f, x, shouldBlock=True):
    # warmup
    y = f(x).block_until_ready() if shouldBlock else f(x)
    # running the code
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    y = f(x).block_until_ready() if shouldBlock else f(x)
    end_wall = time.perf_counter()
    end_cpu = time.process_time()
    # computing the metric and displaying it
    wall_time = end_wall - start_wall
    cpu_time = end_cpu - start_cpu
    cpu_count = os.cpu_count()
    print(
        f"{name}: cpu usage {cpu_time/wall_time:.1f}/{cpu_count} wall_time:{wall_time:.1f}s"
    )


# test functions
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, shape=(5000,), dtype=jnp.float64)
x_mat = jax.random.normal(key, shape=(1000, 1000), dtype=jnp.float64)
f_numpy = np.cos
f_vmap = jax.jit(jax.vmap(jnp.cos))
f_xmap = jax.jit(
    jax.experimental.maps.xmap(jnp.cos, in_axes=[["batch"]], out_axes=["batch"])
)
f_dot = jax.jit(lambda x: jnp.dot(x, x.T))  # to show that JAX can indeed use all cores

timer("numpy", f_numpy, x, shouldBlock=False)
timer("vmap", f_vmap, x)
# timer("xmap", f_xmap, x)
timer("dot", f_dot, x_mat)
