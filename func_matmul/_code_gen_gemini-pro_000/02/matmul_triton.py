# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

# Assume benchmark.py exists in the same directory
# from benchmark import verify_correctness_func, benchmark_performance_func

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'num_stages': 5, 'num_warps': 2}, num_stages=5, num_warps=2),
        # Add more configs or broader ranges if needed
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 4, 'num_warps': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'num_stages': 3, 'num_warps': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'num_stages': 3, 'num_warps': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Hint for compiler: Even K improves performance for some operations
    EVEN_K: tl.constexpr
):
    """
    Description:
        Computes matrix multiplication C = A * B using a tiled approach with grouping.
        Each program instance computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of C.
        Tiles are grouped along the M dimension by GROUP_SIZE_M for potentially better L2 cache locality.

    Args:
        a_ptr (triton.Pointer[dtype]): Pointer to the A matrix in global memory. [Shape: (M, K)]
        b_ptr (triton.Pointer[dtype]): Pointer to the B matrix in global memory. [Shape: (K, N)]
        c_ptr (triton.Pointer[dtype]): Pointer to the C matrix in global memory. [Shape: (M, N)]
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride for the M dimension of matrix A (row stride).
        stride_ak (int): Stride for the K dimension of matrix A (column stride).
        stride_bk (int): Stride for the K dimension of matrix B (row stride).
        stride_bn (int): Stride for the N dimension of matrix B (column stride).
        stride_cm (int): Stride for the M dimension of matrix C (row stride).
        stride_cn (int): Stride for the N dimension of matrix C (column stride).
        BLOCK_SIZE_M (tl.constexpr): Tile size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Tile size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Tile size for the K dimension (inner loop).
        GROUP_SIZE_M (tl.constexpr): Number of tiles grouped along the M dimension.
        EVEN_K (tl.constexpr): Hint indicating if K is even.

    Returns:
        None: The result is written directly to the `c_ptr` tensor.

    Explanation:
        1. Calculate Program IDs: Determine the unique ID (`pid`) for this kernel instance and the total number of instances along M and N dimensions (`num_pid_m`, `num_pid_n`).
        2. Determine Grouping: Calculate the group ID (`group_id`), the starting M-block index for the group (`first_pid_m`), the actual number of M-blocks in this group (`group_size_m_actual`), and the specific M-block index (`pid_m`) this instance is responsible for within its group. This logic distributes blocks considering the `GROUP_SIZE_M`.
        3. Calculate Block Offsets: Compute the starting row offsets (`offs_m`) for the A and C blocks and column offsets (`offs_n`) for the B and C blocks handled by this instance.
        4. Initialize Accumulator: Create a `BLOCK_SIZE_M x BLOCK_SIZE_N` block of zeros (`accumulator`) in registers to store the intermediate dot products. Use `float32` for accumulation for better precision.
        5. Calculate K Offsets: Define the initial offsets (`offs_k`) for the K dimension within a block.
        6. Create Initial Pointers: Calculate the starting memory addresses (`a_ptrs`, `b_ptrs`) for the first blocks of A and B needed for the computation.
        7. Iterate over K Dimension: Loop through the K dimension in steps of `BLOCK_SIZE_K`.
            a. Define K Masks: Create masks (`a_mask_k`, `b_mask_k`) to handle boundary conditions where K is not perfectly divisible by `BLOCK_SIZE_K`.
            b. Load A Block: Load a `BLOCK_SIZE_M x BLOCK_SIZE_K` block from `a_ptr` into registers (`a`). Apply masks (`offs_m_mask`, `a_mask_k`) to avoid out-of-bounds reads, loading 0.0 for masked elements.
            c. Load B Block: Load a `BLOCK_SIZE_K x BLOCK_SIZE_N` block from `b_ptr` into registers (`b`). Apply masks (`b_mask_k`, `offs_n_mask`) similarly.
            d. Compute Dot Product: Perform matrix multiplication (`tl.dot`) on the loaded blocks `a` and `b` and add the result to the `accumulator`.
            e. Advance Pointers: Increment `a_ptrs` and `b_ptrs` to point to the next blocks along the K dimension for the subsequent iteration.
        8. Cast Accumulator: Convert the `accumulator` (in `float32`) to the data type of the output matrix C (`c_ptr.dtype.element_ty`).
        9. Calculate C Pointers: Compute the memory addresses (`c_ptrs`) for the output block in matrix C corresponding to this instance's `offs_m` and `offs_n`.
        10. Store Result: Write the computed block (`c`) to the C matrix in global memory (`c_ptr`). Apply masks (`offs_m_mask`, `offs_n_mask`) to prevent out-of-bounds writes for tiles at the matrix boundaries.
    """
    # 1. Calculate Program IDs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 2. Determine Grouping
    # Grouping logic from Triton tutorial (03-matrix-multiplication.py, grouped)
    # This determines which M, N block this specific program instance handles
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m_actual = min(num_pid_m - first_pid_m, GROUP_SIZE_M) # Handle edge case (M not divisible by BLOCK_SIZE_M * GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m_actual)
    pid_n = (pid % num_pid_in_group) // group_size_m_actual

    # 3. Calculate Block Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m_mask = offs_m < M
    offs_n_mask = offs_n < N

    # 4. Initialize Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 5. Calculate K Offsets
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 6. Create Initial Pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak) # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    # 7. Iterate over K Dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Define K Masks
        k_remaining = K - k * BLOCK_SIZE_K
        a_mask_k = offs_k[None, :] < k_remaining
        b_mask_k = offs_k[:, None] < k_remaining

        # Load A Block
        a = tl.load(a_ptrs, mask=offs_m_mask[:, None] & a_mask_k, other=0.0)
        # Load B Block
        b = tl.load(b_ptrs, mask=b_mask_k & offs_n_mask[None, :], other=0.0)

        # Compute Dot Product
        # Use `allow_tf32=True` for potential speedup on compatible hardware,
        # otherwise matmul precision might be lower than expected.
        # Accumulate in float32
        accumulator += tl.dot(a, b, allow_tf32=True)

        # Advance Pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 8. Cast Accumulator
    # Cast accumulator to the output tensor's dtype
    c = accumulator.to(c_ptr.dtype.element_ty)

    # 9. Calculate C Pointers
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]

    # 10. Store Result
    c_mask = offs_m_mask[:, None] & offs_n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Performs matrix multiplication C = A @ B using the Triton kernel `matmul_kernel`.
        It checks input validity, prepares the output tensor, calculates the grid,
        and launches the autotuned Triton kernel.

    Args:
        A (torch.Tensor): The first input matrix. [Shape: (M, K)]
        B (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The result matrix C. [Shape: (M, N)]

    Explanation:
        1. Input Validation: Check if A and B are 2D tensors, reside on the same CUDA device,
           have compatible data types (float16, bfloat16, or float32), and compatible dimensions for multiplication (A.shape[1] == B.shape[0]).
           Ensure tensors are contiguous for optimal memory access.
        2. Get Dimensions: Extract M, K from A.shape and N from B.shape.
        3. Create Output Tensor: Allocate the output tensor `C` with shape (M, N) on the same device and with the same dtype as A.
        4. Define Grid Calculation: Create a lambda function `grid` that determines the launch grid size for the kernel. The grid is 1D, and its size is the total number of M-N blocks required, calculated as `ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)`. The kernel internally uses this 1D `pid` and `GROUP_SIZE_M` to map to the correct 2D block coordinates.
        5. Launch Kernel: Call the `matmul_kernel` via the autotuner.
           - Pass the pointers to A, B, and C tensors.
           - Pass the dimensions M, N, K.
           - Pass the strides for each dimension of A, B, and C.
           - The `grid` function provides the launch configuration.
           - `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M` are automatically filled by the autotuner.
           - `EVEN_K` hint is passed for potential optimization.
        6. Return Result: Return the output tensor `C` containing the matrix multiplication result.
    """
    # 1. Input Validation
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA device"
    assert A.device == B.device, "Input tensors must be on the same device"
    # assert A.dtype == B.dtype, "Input tensors must have the same dtype" # Can relax if kernel handles mixed precision
    assert A.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert B.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    assert A.dim() == 2 and B.dim() == 2, "Input matrices must be 2D"
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Incompatible dimensions: A.shape={A.shape}, B.shape={B.shape}"

    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()

    # 3. Create Output Tensor
    # Allocate output tensor C on the same device and with compatible dtype
    # If accumulating in float32, output dtype should ideally match inputs or be float32
    output_dtype = A.dtype # Or choose based on desired precision, e.g., torch.float32
    C = torch.empty((M, N), device=A.device, dtype=output_dtype)

    # 4. Define Grid Calculation
    # This grid calculation works with the grouping logic inside the kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # 5. Launch Kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        EVEN_K=(K % 2 == 0),
        # BLOCK_SIZE_M, N, K, GROUP_SIZE_M are set by autotuner
    )

    # 6. Return Result
    return C

def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Performs matrix multiplication C = A @ B using PyTorch's native `torch.matmul`.
        This serves as the reference implementation for correctness and performance comparison.

    Args:
        A (torch.Tensor): The first input matrix. [Shape: (M, K)]
        B (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The result matrix C. [Shape: (M, N)]

    Explanation:
        1. Call `torch.matmul(A, B)` which computes the matrix product of A and B.
        2. Return the resulting tensor.
    """
    # 1. Call torch.matmul
    result = torch.matmul(A, B)
    # 2. Return result
    return result

# Main execution block for benchmarking
if __name__ == "__main__":
    # Define matrix size ranges for testing
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]

    # Input generator function
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        """Generates random normalized matrices A and B."""
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1) # Normalize A for stability/consistency
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1) # Normalize B
        return A, B

    # Import benchmarking functions from benchmark.py
    # Ensure benchmark.py is in the PYTHONPATH or the same directory
    try:
        from benchmark import verify_correctness_func, benchmark_performance_func
    except ImportError:
        print("Error: benchmark.py not found.")
        print("Please ensure benchmark.py and _benchmark_util.py are in the same directory or your PYTHONPATH.")
        exit(1)

    # Assign generated functions for benchmarking
    generated_triton_func = matmul_triton
    generated_torch_func = matmul_torch

    # Verify correctness against PyTorch baseline
    print("Verifying correctness...")
    correctness_results = verify_correctness_func(
        generated_triton_func,
        generated_torch_func,
        input_generator,
        [M_sizes, N_sizes, K_sizes], # Pass lists directly
        dtype=torch.float16 # Specify dtype, change if needed
    )
    print(f"Correctness check results: {correctness_results['passed']} passed, {correctness_results['failed']} failed.")
    # Optional: Add assertion or exit if failed
    # assert correctness_results['failed'] == 0, "Correctness check failed!"

    # Benchmark performance against PyTorch baseline (torch.compile'd)
    print("\nBenchmarking performance...")
    median_triton, median_pytorch, median_speedup, performance_results = benchmark_performance_func(
        generated_triton_func,
        generated_torch_func,
        input_generator,
        [M_sizes, N_sizes, K_sizes], # Pass lists directly
        dtype=torch.float16 # Specify dtype, change if needed
    )
    print(f"\nMedian Triton time: {median_triton:.4f} ms")
    print(f"Median PyTorch (compiled) time: {median_pytorch:.4f} ms")
    print(f"Median Speedup (Triton vs PyTorch): {median_speedup:.2f}x")

    print("\nBenchmarking complete. Results saved to JSON files and plot generated (code.performance.func.png).")

