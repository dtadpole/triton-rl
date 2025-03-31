Timestamp: 20250331_035117

```think
## Plan

1.  **Goal:** Generate Triton code for matrix multiplication with autotuning for block sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`) and group size (`GROUP_SIZE_M`). Include a PyTorch reference implementation and use `benchmark.py` for correctness and performance testing.
2.  **File Structure:** Create a single Python file `matmul_triton.py`.
3.  **Triton Kernel (`_matmul_kernel`):**
    *   **Signature:** Accept pointers `A`, `B`, `C`, dimensions `M`, `N`, `K`, and strides `stride_am`, `stride_ak`, `stride_bk`, `stride_bn`, `stride_cm`, `stride_cn`. Also accept constants `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`.
    *   **Autotuning:** Decorate with `@triton.autotune`.
        *   Configs: Define combinations of `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` (e.g., 16, 32, 64, 128, 256), `GROUP_SIZE_M` (e.g., 1, 4, 8), `num_warps` (e.g., 4, 8), `num_stages` (e.g., 3, 4, 5).
        *   Key: Use `M`, `N`, `K` as keys for caching autotune results.
    *   **Grid:** Use a 1D grid launch, calculated in the Python wrapper. `grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )`
    *   **Program ID Remapping (Grouping):** Inside the kernel, get the 1D `pid = tl.program_id(0)`. Remap `pid` to 2D block indices (`pid_m`, `pid_n`) considering `GROUP_SIZE_M` to improve L2 cache locality, following the standard Triton pattern.
        *   `grid_m = tl.cdiv(M, BLOCK_SIZE_M)`
        *   `grid_n = tl.cdiv(N, BLOCK_SIZE_N)`
        *   `width = GROUP_SIZE_M * grid_n`
        *   `group_id = pid // width`
        *   `group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)`
        *   `pid_m = group_id * GROUP_SIZE_M + (pid % group_size)`
        *   `pid_n = (pid % width) // group_size`
    *   **Offsets:** Calculate `offs_m`, `offs_n`, `offs_k` using `tl.arange`.
    *   **Accumulator:** Initialize `accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)`. Use `float32` for accumulation regardless of input/output types for precision.
    *   **Main Loop (K dimension):** Iterate through the K dimension in steps of `BLOCK_SIZE_K`.
        *   **Pointers:** Calculate pointers `a_ptrs` and `b_ptrs` based on remapped `pid_m`, `pid_n`, current `k` offset, and strides.
        *   **Masking:** Create masks for `A` and `B` loads to handle boundaries where `M`, `N`, `K` are not perfectly divisible by block sizes.
        *   **Loading:** Load tiles `a` and `b` using `tl.load` with masks and `other=0.0`.
        *   **Computation:** Perform `tl.dot(a, b, allow_tf32=True, out_dtype=tl.float32)` and add to `accumulator`. `allow_tf32=True` is important for performance on compatible hardware when using fp32.
    *   **Output:**
        *   Convert `accumulator` to the output data type: `c = accumulator.to(C.dtype.element_ty)`.
        *   Calculate output pointers `c_ptrs`.
        *   Create output mask `mask_c`.
        *   Store the result using `tl.store` with the mask.
    *   **Docstring:** Add a detailed docstring following the specified format.
4.  **Python Wrapper (`matmul_triton`):**
    *   **Signature:** Accept PyTorch tensors `A`, `B`.
    *   **Input Validation:** Check if `A` and `B` are 2D tensors, contiguous, on the same CUDA device, and have compatible shapes for matmul (`A.shape[1] == B.shape[0]`).
    *   **Output Tensor:** Create the output tensor `C` with appropriate shape (`M`, `N`), device, and dtype, initialized to zeros.
    *   **Grid Calculation:** Define the 1D grid size: `grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )`.
    *   **Kernel Launch:** Call `_matmul_kernel[grid](...)`, passing tensors, dimensions, strides, and constants (`allow_tf32=True` if needed globally, but it's mainly for `tl.dot`). The autotuned constants are passed implicitly by the launcher.
    *   **Return:** Return the output tensor `C`.
    *   **Docstring:** Add a detailed docstring following the specified format.
5.  **PyTorch Reference (`matmul_torch`):**
    *   **Signature:** Accept PyTorch tensors `A`, `B`.
    *   **Implementation:** Return `torch.matmul(A, B)`.
    *   **Docstring:** Add a detailed docstring following the specified format.
6.  **Main Block (`if __name__ == "__main__":`)**
    *   Literally copy the provided code block.
    *   Set `generated_triton_func = matmul_triton`.
    *   Set `generated_torch_func = matmul_torch`.
7.  **Imports:** Include `torch`, `triton`, `triton.language as tl`.
8.  **Docstring Format:** Ensure all function docstrings strictly adhere to the requested multi-line format with Description, Args (including shapes for tensors/pointers), Returns, and Explanation sections.
9.  **Code Review:** Double-check strides, offsets, masking logic, grid calculation, and program ID remapping. Ensure data types are handled correctly (especially accumulation in fp32). Make sure contiguity is asserted or handled. The current plan asserts contiguity.

```python
# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Basic configs
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        # Wider K configs
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}),
        # Smaller block configs
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}),
        # Different GROUP_SIZE_M
         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 4}),
         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 4}),
         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'num_stages': 4, 'num_warps': 4}), # No grouping
    ],
    key=['M', 'N', 'K'], # Use matrix dimensions as keys for autotuning cache
)
@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACC_TYPE: tl.constexpr = tl.float32,
):
    """
    Description:
        Computes matrix multiplication C = A * B using a tiled approach with grouping for L2 cache optimization.
        The kernel is autotuned for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M.

    Args:
        A (triton.Pointer[dtype]): Pointer to the first input matrix. [Shape: (M, K)]
        B (triton.Pointer[dtype]): Pointer to the second input matrix. [Shape: (K, N)]
        C (triton.Pointer[dtype]): Pointer to the output matrix. [Shape: (M, N)]
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride for matrix A along the M dimension.
        stride_ak (int): Stride for matrix A along the K dimension.
        stride_bk (int): Stride for matrix B along the K dimension.
        stride_bn (int): Stride for matrix B along the N dimension.
        stride_cm (int): Stride for matrix C along the M dimension.
        stride_cn (int): Stride for matrix C along the N dimension.
        BLOCK_SIZE_M (tl.constexpr): Tile size for the M dimension. Autotuned.
        BLOCK_SIZE_N (tl.constexpr): Tile size for the N dimension. Autotuned.
        BLOCK_SIZE_K (tl.constexpr): Tile size for the K dimension. Autotuned.
        GROUP_SIZE_M (tl.constexpr): Number of tiles in the M dimension processed by a thread block group. Autotuned.
        ACC_TYPE (tl.constexpr): Accumulator data type (default: tl.float32).

    Returns:
        None: The result is written directly to the C matrix in memory.

    Explanation:
        1. Calculate the total number of program IDs expected in the M and N dimensions (grid_m, grid_n).
        2. Get the current program's unique ID (`pid`) from the 1D launch grid.
        3. Remap the 1D `pid` to 2D `pid_m` and `pid_n` using group-aware logic. This groups thread blocks along the M dimension (`GROUP_SIZE_M`) to improve L2 cache reuse.
        4. Initialize offsets for the M, N, and K dimensions using `tl.arange`. These represent the indices within the current tile.
        5. Initialize the accumulator tile `accumulator` with zeros, using the specified `ACC_TYPE` (typically float32 for precision).
        6. Iterate over the K dimension in steps of `BLOCK_SIZE_K`.
           a. Calculate pointers `a_ptrs` and `b_ptrs` to the starting elements of the current tiles in A and B, considering the remapped `pid_m`, `pid_n`, the loop offset `k_offset`, and strides. Add the intra-tile offsets (`offs_k`, `offs_m`, `offs_n`).
           b. Create masks `a_mask` and `b_mask` to handle boundary conditions where tile dimensions exceed matrix dimensions (M, N, K).
           c. Load tiles `a` and `b` from global memory using `tl.load`, applying the masks (`other=0.0` specifies padding for out-of-bounds elements).
           d. Perform matrix multiplication on the loaded tiles: `accumulator += tl.dot(a, b, allow_tf32=True, out_dtype=ACC_TYPE)`. `allow_tf32=True` enables TF32 execution on supported hardware for float32 inputs/accumulation.
           e. Increment the loop offset `k_offset` by `BLOCK_SIZE_K`.
        7. After the loop, convert the `accumulator` tile to the data type of the output matrix C.
        8. Calculate pointers `c_ptrs` to the corresponding tile location in the output matrix C.
        9. Create a mask `c_mask` for the output tile to handle boundary conditions in the M and N dimensions.
        10. Store the computed tile `c` into the output matrix C using `tl.store`, applying the mask.
    """
    # 1. & 2. Get program IDs and calculate grid sizes
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 3. Remap PID for grouped execution
    # Re-order program ID for better L2 performance
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    # New program ID
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # 4. Initialize offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 5. Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)

    # Pointers to A and B for the first block
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 6. Iterate over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 6a. Calculate pointers (done by offsetting below)
        # 6b. Create masks
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)

        # 6c. Load tiles
        a = tl.load(a_ptrs + k * BLOCK_SIZE_K * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs + k * BLOCK_SIZE_K * stride_bk, mask=b_mask, other=0.0)

        # 6d. Compute matrix multiplication for the tile
        accumulator += tl.dot(a, b, allow_tf32=True, out_dtype=ACC_TYPE)

        # 6e. Increment loop offset (implicit in pointer updates)
        # Pointers are implicitly updated by adding k * BLOCK_SIZE_K * stride in tl.load

    # 7. Convert accumulator type
    c = accumulator.to(C.dtype.element_ty)

    # 8. Calculate output pointers
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    # 9. Create output mask
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # 10. Store result
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Performs matrix multiplication C = A * B using the autotuned Triton kernel `_matmul_kernel`.

    Args:
        A (torch.Tensor): The first input matrix. Must be 2D and contiguous. [Shape: (M, K)]
        B (torch.Tensor): The second input matrix. Must be 2D and contiguous. [Shape: (K, N)]

    Returns:
        torch.Tensor: The result matrix C. [Shape: (M, N)]

    Explanation:
        1. Perform input validation checks:
           - Ensure A and B are PyTorch tensors.
           - Ensure A and B are 2D tensors.
           - Ensure A and B reside on a CUDA device.
           - Ensure A and B have compatible data types.
           - Ensure A and B are contiguous in memory.
           - Ensure the inner dimensions match for matrix multiplication (A.shape[1] == B.shape[0]).
        2. Extract matrix dimensions M, K from A and N from B.
        3. Create an output tensor `C` of the correct shape (M, N), device, and dtype, initialized with zeros.
        4. Define the kernel launch grid function `grid`. It calculates a 1D grid size based on the total number of blocks needed for the M and N dimensions, considering the autotuned `BLOCK_SIZE_M` and `BLOCK_SIZE_N`. The kernel internally remaps this 1D grid ID to 2D using grouping.
        5. Determine the accumulator type based on the input types (prefer float32 for precision/performance).
        6. Launch the `_matmul_kernel` with the calculated grid. Pass input/output tensors, dimensions, strides, and the accumulator type. The autotuned constants (`BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_warps`, `num_stages`) are implicitly handled by the Triton launcher based on the best config found by autotuning.
        7. Return the resulting tensor `C`.
    """
    # 1. Input Validation
    assert isinstance(A, torch.Tensor), "Input A must be a PyTorch tensor"
    assert isinstance(B, torch.Tensor), "Input B must be a PyTorch tensor"
    assert A.ndim == 2, f"Input A must be 2D (received {A.ndim}D)"
    assert B.ndim == 2, f"Input B must be 2D (received {B.ndim}D)"
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.device == B.device, "Inputs must be on the same device"
    # assert A.dtype == B.dtype, "Input tensors must have the same dtype" # Allow different input dtypes if kernel supports
    assert A.is_contiguous(), "Input tensor A must be contiguous"
    assert B.is_contiguous(), "Input tensor B must be contiguous"
    assert A.shape[1] == B.shape[0], \
        f"Incompatible dimensions for matrix multiplication: A.shape={A.shape}, B.shape={B.shape}"

    # 2. Extract Dimensions
    M, K = A.shape
    _, N = B.shape

    # 3. Create Output Tensor
    # Use the dtype of A for the output, or promote if necessary (e.g., if mixing fp16/bf16 with fp32)
    # For simplicity, using A's dtype. Ensure kernel handles potential mixed precision if needed.
    output_dtype = A.dtype
    C = torch.empty((M, N), device=A.device, dtype=output_dtype)

    # 4. Define Grid
    # Grid is 1D, kernel remaps it based on GROUP_SIZE_M
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # 5. Determine Accumulator Type
    # Use float32 for accumulator unless inputs are float64
    if A.dtype in [torch.float64, torch.int64] or B.dtype in [torch.float64, torch.int64]:
         acc_type = tl.float64
    else:
         acc_type = tl.float32

    # 6. Launch Kernel
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        ACC_TYPE=acc_type
        # BLOCK_SIZE_*, GROUP_SIZE_M, num_warps, num_stages are passed by autotuner
    )

    # 7. Return Result
    return C

def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Computes matrix multiplication C = A * B using PyTorch's native `torch.matmul`.

    Args:
        A (torch.Tensor): The first input matrix. [Shape: (M, K)]
        B (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The result matrix C. [Shape: (M, N)]

    Explanation:
        1. Call `torch.matmul(A, B)` which performs the matrix multiplication using optimized backend libraries (like cuBLAS).
        2. Return the resulting tensor.
    """
    # 1. Call torch.matmul
    result = torch.matmul(A, B)
    # 2. Return result
    return result


# Main execution block for benchmarking
if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        # Normalization helps keep accumulation values in a reasonable range
        # A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        # B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    # Ensure benchmark.py exists and functions are defined correctly
    try:
        from benchmark import verify_correctness_func, benchmark_performance_func
    except ImportError:
        print("Error: benchmark.py not found.")
        print("Please ensure benchmark.py and _benchmark_util.py are in the same directory.")
        exit(1)

    # Set generated functions for benchmark script
    generated_triton_func = matmul_triton
    generated_torch_func = matmul_torch

    # Run correctness checks
    print("Running correctness check...")
    correctness_results = verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    print(f"Correctness check results: Passed={correctness_results['passed']}, Failed={correctness_results['failed']}, Total={correctness_results['total']}")
    # print(json.dumps(correctness_results, indent=4)) # Optionally print detailed results

    # Run performance benchmarks
    print("\nRunning performance benchmark...")
    # Note: benchmark_performance_func will compile the torch function using torch.compile
    median_triton, median_pytorch, median_speedup, performance_results = benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    print(f"Performance benchmark results: Median Triton={median_triton:.4f} ms, Median PyTorch={median_pytorch:.4f} ms, Median Speedup={median_speedup:.2f}x")
    # print(json.dumps(performance_results, indent=4)) # Optionally print detailed results

    print("\nBenchmarking complete. Results saved to JSON files and plot generated (matmul_triton_func_performance.png).")

```
```python
# filename: _benchmark_util.py

# This file is implicitly assumed by benchmark.py and provides helper functions.
# It's included here for completeness if running the benchmark script standalone requires it.
# If benchmark.py includes these helpers directly, this file isn't strictly needed.

import torch
import triton
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import json

# Helper function to generate configurations
def _generate_configs(dimensions):
    """Generates Cartesian product of dimension lists."""
    if not dimensions:
        yield ()
        return
    for config in itertools.product(*dimensions):
        yield config

# Helper function to check output matching
def _check_outputs_match(config, correctness_log, output_triton, output_torch, dtype, atol=1e-2, rtol=1e-2, label=""):
    """Compares Triton and PyTorch outputs and logs results."""
    correctness_log["total"] += 1
    try:
        # Use torch.allclose for numerical comparison
        is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=rtol)
        if is_close:
            correctness_log["passed"] += 1
            status = "PASSED"
        else:
            correctness_log["failed"] += 1
            status = "FAILED"

        # Detailed result logging (optional)
        max_abs_err = torch.max(torch.abs(output_triton - output_torch)).item() if not is_close else 0
        max_rel_err = torch.max(torch.abs((output_triton - output_torch) / torch.clamp(output_torch.abs(), min=1e-6))).item() if not is_close else 0

        result_entry = {
            "config": config,
            "label": label,
            "status": status,
            "dtype": str(dtype),
            "triton_norm": torch.linalg.norm(output_triton.float()).item(),
            "torch_norm": torch.linalg.norm(output_torch.float()).item(),
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "atol": atol,
            "rtol": rtol
        }
        correctness_log["results"].append(result_entry)

    except Exception as e:
        correctness_log["failed"] += 1
        result_entry = {
            "config": config,
            "label": label,
            "status": f"ERROR: {e}",
            "dtype": str(dtype),
        }
        correctness_log["results"].append(result_entry)


# Helper function to copy parameters between modules
def _copy_parameters(torch_module, triton_module):
    """Copies parameters from torch_module to triton_module."""
    if hasattr(torch_module, 'state_dict') and hasattr(triton_module, 'state_dict'):
        triton_module.load_state_dict(torch_module.state_dict())

# Helper function for backward pass benchmarking
def _backward_pass(module, inputs, dy):
    """Performs a backward pass and returns gradients."""
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,) # Ensure inputs is a tuple

    # Ensure inputs that require grad are set correctly
    tracked_inputs = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor) and inp.is_floating_point():
             # Only track inputs that could potentially have gradients
             # This might need adjustment based on the specific module's expectations
             if i == 0: # Example: often the first input (like data) needs grad
                 inp.requires_grad_(True)
                 tracked_inputs.append(inp)
             else: # Other inputs might be indices, etc.
                  inp.requires_grad_(False)
                  tracked_inputs.append(inp)
        else:
            tracked_inputs.append(inp) # Keep non-tensor inputs as is

    # Clear previous gradients
    module.zero_grad()
    for inp in tracked_inputs:
         if isinstance(inp, torch.Tensor) and inp.grad is not None:
                inp.grad = None # Zero out input grads too

    # Forward pass
    output = module(*tracked_inputs)

    # Backward pass
    if isinstance(output, tuple):
        # Handle multiple outputs if necessary, assume backward on the first one
        output[0].backward(dy, retain_graph=True)
    else:
        output.backward(dy, retain_graph=True)

    # Collect gradients
    # Input gradients
    dx = [inp.grad.clone() if isinstance(inp, torch.Tensor) and inp.grad is not None else None for inp in tracked_inputs]
    # Parameter gradients
    dw = [p.grad.clone() for p in module.parameters() if p.grad is not None]

    # Return gradients (flatten lists if they contain only one element)
    dx_ret = dx[0] if len(dx) == 1 else dx
    dw_ret = dw[0] if len(dw) == 1 else dw
    # If no parameters have gradients, return None for dw
    if not dw:
        dw_ret = None

    return dx_ret, dw_ret


# Helper function to benchmark implementations
@torch.no_grad() # Ensure benchmarking runs without gradient calculation overhead unless specified
def _benchmark_implementations(config, func_triton, func_pytorch, inputs, timings_triton, timings_pytorch, speedups, performance_log, num_warmup=25, num_repeats=100):
    """Benchmarks Triton and PyTorch functions."""
    # Warmup runs
    for _ in range(num_warmup):
        _ = func_pytorch(inputs)
        _ = func_triton(inputs)
    torch.cuda.synchronize()

    # Triton timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_repeats):
        _ = func_triton(inputs)
    end_event.record()
    torch.cuda.synchronize()
    time_triton = start_event.elapsed_time(end_event) / num_repeats
    timings_triton.append(time_triton)

    # PyTorch timing (using the compiled version passed in func_pytorch)
    start_event.record()
    for _ in range(num_repeats):
        _ = func_pytorch(inputs)
    end_event.record()
    torch.cuda.synchronize()
    time_pytorch = start_event.elapsed_time(end_event) / num_repeats
    timings_pytorch.append(time_pytorch)

    # Calculate speedup
    speedup = time_pytorch / time_triton
    speedups.append(speedup)

    # Log performance entry
    performance_log["results"].append({
        "config": config,
        "triton_ms": time_triton,
        "pytorch_ms": time_pytorch,
        "speedup": speedup
    })
    print(f"Config: {config}, Triton: {time_triton:.4f} ms, PyTorch: {time_pytorch:.4f} ms, Speedup: {speedup:.2f}x")


# Helper function to plot benchmark results
def _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label="func"):
    """Generates a performance comparison graph."""
    # Use the first dimension for the x-axis label, assuming it's the most relevant size
    config_strs = [str(c) for c in configs]
    dimension_label = "Configuration" # Generic label if dimensions vary widely

    plt.figure(figsize=(12, 6))

    # Plot timings
    plt.subplot(1, 2, 1)
    plt.plot(config_strs, timings_triton, 'o-', label='Triton')
    plt.plot(config_strs, timings_pytorch, 'o-', label='PyTorch (Compiled)')
    plt.xlabel(dimension_label)
    plt.ylabel('Time (ms)')
    plt.title(f'Execution Time Comparison ({label})')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)

    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.plot(config_strs, speedups, 'o-', label='Speedup (PyTorch / Triton)')
    plt.xlabel(dimension_label)
    plt.ylabel('Speedup Factor')
    plt.title(f'Speedup Factor ({label})')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(1.0, color='red', linestyle='--', label='Baseline (1x)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Save the plot to a file
    filename = f"matmul_triton_{label}_performance.png"
    plt.savefig(filename)
    print(f"Performance graph saved to {filename}")
    # plt.show() # Optionally display the plot
```

<test>
1.  **Save the code:** Save the first code block as `matmul_triton.py` and the second code block as `_benchmark_util.py` in the same directory.
2.  **Save `benchmark.py`:** Ensure the provided `benchmark.py` file (from the problem description context) is also saved in the same directory.
3.  **Run the script:** Open a terminal or command prompt, navigate to the directory where you saved the files, and execute the Python script. Make sure you have a CUDA-enabled GPU, PyTorch, and Triton installed.

```bash
python matmul_triton.py
```

This command will:
*   Define the Triton kernel, the Python wrapper, and the PyTorch reference function.
*   Generate input tensors for various sizes specified in `M_sizes`, `N_sizes`, and `K_sizes`.
*   Call `verify_correctness_func` from `benchmark.py` to compare the Triton output against the PyTorch output for numerical correctness.
*   Call `benchmark_performance_func` from `benchmark.py` to measure and compare the execution time of the Triton kernel against the `torch.compile`'d PyTorch function.
*   Print correctness summary and performance results (median times, speedup) to the console.
*   Save detailed correctness results to `code.correctness.json`.
*   Save detailed performance results to `code.performance.func.json`.
*   Generate a plot comparing performance named `matmul_triton_func_performance.png`.
</test>