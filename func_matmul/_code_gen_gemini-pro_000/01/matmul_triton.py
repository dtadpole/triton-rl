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

