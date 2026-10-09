import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:
        left = d_in[idx - 1] if idx > 0 else d_in[0]
        center = d_in[idx]
        right = d_in[idx + 1] if idx < N - 1 else d_in[N - 1]

        d_out[idx] = 0.25 * left + 0.5 * center + 0.25 * right


def run_stencil(h_in):
    N = len(h_in)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)

    threads = 256
    blocks = (N + threads - 1) // threads

    stencil_1d[blocks, threads](d_in, d_out, N)
    cuda.synchronize()

    return d_out.copy_to_host()


def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode='edge')

    return (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )


if __name__ == "__main__":
    N = 10007

    h_input = np.sin(
        np.linspace(0, 10, N)
    ).astype(np.float32)

    h_out_gpu = run_stencil(h_input)
    cpu_ref = cpu_stencil(h_input)

    max_delta = np.max(np.abs(h_out_gpu - cpu_ref))

    assert np.allclose(h_out_gpu, cpu_ref, atol=1e-4)

    print("TASK 2 PASSED: MAX DELTA =", max_delta)
