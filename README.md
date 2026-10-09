# CUDA Lab 02

**Name:** Nuriya Sultanseitova  
**Student ID:** 230103360  
**Course:** CSS 314 - Parallel Computing

## Environment

I used Google Colab with an NVIDIA Tesla T4 GPU to complete this lab. The code was written in Python using Numba CUDA and NumPy.

- GPU: Tesla T4
- Compute Capability: 7.5
- Threads per block (Task 1): 256

## Task 1 - Warp Divergence

In this task, I tested three different CUDA kernels to see how warp divergence affects execution time.

My results:

| Kernel | Time |
|---|---|
| A - Uniform | 24.2959 ms |
| B - Divergent | 95.6944 ms |
| C - Warp-Aligned | 47.7439 ms |

Kernel B was about 3.94 times slower than Kernel A. Kernel C was faster than B, but still slower than A.

I learned that when threads in the same warp follow different branches, GPU performance can become slower.

## Task 2 - 1D Stencil

For this task, I implemented a 1D stencil operation and compared the GPU output with the CPU result.

- Array size: 10007
- Maximum difference: 5.9604645e-08
- Result: Passed

The results were very close, so the GPU calculation worked correctly for my test.

## Task 3 - Grid-Stride Loop

In Task 3, I used a grid-stride loop to multiply all elements of a large array by 4.25.

- Array size: 16777216
- Factor: 4.25
- Result: Passed

I checked the first and last elements of the array, and all tested values were correct.

## Task 4 - 2D Sobel Filter

For the last task, I implemented a Sobel-X filter using a 2D CUDA grid.

- Image size: 2048 x 2048
- Block size: 16 x 16
- Maximum absolute value: 0.0
- Result: Passed

I tested the kernel using an image filled with ones. The output was zero, as expected, because there were no intensity changes.

## Verification

I ran `verify_submission.py` to check Tasks 2, 3, and 4.

Results:

- Task 2: PASS
- Task 3: PASS
- Task 4: PASS

Verification token: `1034DE005B8A90C9F512`

## Files

- `task1_divergence.py`
- `task2_stencil_1d.py`
- `task3_grid_stride.py`
- `task4_sobel_2d.py`
- `verify_submission.py`

All four tasks were completed and tested in Google Colab.
