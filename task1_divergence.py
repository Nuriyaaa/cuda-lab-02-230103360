import numpy as np
import time
from numba import cuda

N = 2**20
THREADS = 256

@cuda.jit
def kernel_a(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        value = d_in[idx]
        for i in range(1000):
            value = value * 1.000001 + 0.000001
        d_out[idx] = value

@cuda.jit
def kernel_b(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        value = d_in[idx]
        if idx % 2 == 0:
            for i in range(1000):
                value = value * 1.000001 + 0.000001
        else:
            for i in range(1000):
                value = (value - 0.000001) / 1.000001
        d_out[idx] = value

@cuda.jit
def kernel_c(d_in, d_out, N):
    idx = cuda.grid(1)
    if idx < N:
        value = d_in[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for i in range(1000):
                value = value * 1.000001 + 0.000001
        else:
            for i in range(1000):
                value = (value - 0.000001) / 1.000001
        d_out[idx] = value

def benchmark(kernel, d_in, d_out, blocks):
    kernel[blocks, THREADS](d_in, d_out, N)
    cuda.synchronize()

    times = []
    for _ in range(10):
        start = time.perf_counter()
        kernel[blocks, THREADS](d_in, d_out, N)
        cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times)

def main():
    h_arr = np.ones(N, dtype=np.float32)
    d_in = cuda.to_device(h_arr)
    d_out = cuda.device_array_like(d_in)
    blocks = (N + THREADS - 1) // THREADS

    for name, kernel in [
        ("Kernel A - Uniform", kernel_a),
        ("Kernel B - Divergent", kernel_b),
        ("Kernel C - Warp-Aligned", kernel_c)
    ]:
        avg = benchmark(kernel, d_in, d_out, blocks)
        print(f"{name}: {avg:.4f} ms")

if __name__ == "__main__":
    main()
