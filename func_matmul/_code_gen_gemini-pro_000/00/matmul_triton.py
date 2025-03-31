# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Basic configurations
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Larger block sizes
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Different group sizes
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'], # Key parameters for caching autotuning results
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,            # Pointers to matrices A, B, and C
    M, N, K,                        # Matrix dimensions
    stride_am, stride_ak,           # Strides for matrix A
    stride_bk, stride_bn,           # Strides for matrix B
    stride_cm, stride_cn,           # Strides for matrix C
    BLOCK_SIZE_M: tl.constexpr,     # Tile size for M dimension
    BLOCK_SIZE_N: tl.constexpr,     # Tile size for N dimension
    BLOCK_SIZE_K: tl.constexpr,     # Tile size for K dimension
    GROUP_SIZE_M: tl.constexpr      # Number of M-blocks processed by a group of programs
):
    """
    Description:
        Computes matrix multiplication C = A @ B using a blocked algorithm with grouping.
        Each program instance computes a block of C. Blocks are grouped along the M dimension
        for potentially better L2 cache locality.

    Args:
        A_ptr (triton.language.pointer): Pointer to the input matrix A. [Shape: (M, K)]
        B_ptr (triton.language.pointer): Pointer to the input matrix B. [Shape: (K, N)]
        C_ptr (triton.language.pointer): Pointer to the output matrix C. [Shape: (M, N)]
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride for the M dimension of matrix A (bytes per row).
        stride_ak (int): Stride for the K dimension of matrix A (bytes per element in a row).
        stride_bk (int): Stride for the K dimension of matrix B (bytes per row).
        stride_bn (int): Stride for the N dimension of matrix B (bytes per element in a row).
        stride_cm (int): Stride for the M dimension of matrix C (bytes per row).
        stride_cn (int): Stride for the N dimension of matrix C (bytes per element in a row).
        BLOCK_SIZE_M (tl.constexpr): The height of the tile computed by each thread block.
        BLOCK_SIZE_N (tl.constexpr): The width of the tile computed by each thread block.
        BLOCK_SIZE_K (tl.constexpr): The size of the inner dimension block used for loading A and B.
        GROUP_SIZE_M (tl.constexpr): The number of blocks along the M dimension handled by a workgroup.

    Returns:
        None: The result is written directly to the C_ptr memory location.

    Explanation:
        1. Calculate Program IDs: Determine the unique ID (`pid`) for the current program instance. Split this ID into M (`pid_m`) and N (`pid_n`) components based on the total number of blocks needed.
        2. Grouping Logic: Calculate the group ID (`group_id`) and the starting M-block ID (`first_pid_m`) for the group this program belongs to. Determine the actual number of M-blocks this group handles (`group_size`), considering boundary conditions.
        3. Initialize Offsets: Compute the starting row offsets (`offs_m`) and column offsets (`offs_n`) for the C tile this specific program instance will compute, accounting for its position within the group.
        4. Initialize Accumulator: Create a `BLOCK_SIZE_M x BLOCK_SIZE_N` accumulator tile (`accumulator`) initialized to zeros. Use `float32` for accumulation to maintain precision, regardless of input types.
        5. Initialize A and B Pointers: Calculate the initial pointers for the A and B tiles based on the program's M/N offsets (`offs_m`, `offs_n`) and the starting K offset (0).
        6. Loop Over K Dimension: Iterate through the K dimension in steps of `BLOCK_SIZE_K`.
            a. Load A Tile: Define pointers for the `BLOCK_SIZE_M x BLOCK_SIZE_K` tile of A. Load the tile using `tl.load`, applying masking (`mask=offs_k[:, None] < K`) to handle incomplete tiles at the K boundary.
            b. Load B Tile: Define pointers for the `BLOCK_SIZE_K x BLOCK_SIZE_N` tile of B. Load the tile using `tl.load`, applying masking (`mask=offs_k[None, :] < K`) for the K boundary.
            c. Matrix Multiply Tile: Compute the dot product of the loaded A and B tiles (`tl.dot(a, b)`). The `allow_tf32` argument enables TensorFloat-32 for faster computation on supported hardware.
            d. Accumulate Result: Add the result of the tile multiplication to the `accumulator`.
            e. Update Pointers: Advance the A and B pointers along the K dimension by `BLOCK_SIZE_K`.
        7. Cast Accumulator: After the K loop finishes, cast the `accumulator` (which is `float32`) to the data type of the output matrix C (`C_ptr.dtype.element_ty`).
        8. Calculate C Pointers: Define the pointers (`c_ptrs`) for the `BLOCK_SIZE_M x BLOCK_SIZE_N` output tile in C based on the calculated offsets (`offs_m`, `offs_n`).
        9. Store Result: Write the computed tile (`c`) to the C matrix using `tl.store`. Apply masking (`mask=mask`) to prevent out-of-bounds writes for tiles at the M and N boundaries of the matrix.
    """
    # 1. Calculate Program IDs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # 2. Grouping Logic
    # Determine the M and N block indices for this program
    # If GROUP_SIZE_M is 1, this simplifies to standard block calculation
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # Within the group, determine the specific M and N block indices
    pid_m = first_pid_m + (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n

    # 3. Initialize Offsets
    # Calculate the starting row indices for the M block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Calculate the starting column indices for the N block
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # Calculate the starting K indices (innermost loop)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 4. Initialize Accumulator
    # `accumulator` holds the `BLOCK_SIZE_M x BLOCK_SIZE_N` tile results
    # Use float32 for accumulation for better precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 5. Initialize A and B Pointers
    # Pointer for the A tile: starts at row `offs_m`, column `offs_k`
    A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # Pointer for the B tile: starts at row `offs_k`, column `offs_n`
    B_tile_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 6. Loop Over K Dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Define boundary masks for K dimension
        k_boundary_mask_a = (offs_k[None, :] + k * BLOCK_SIZE_K) < K
        k_boundary_mask_b = (offs_k[:, None] + k * BLOCK_SIZE_K) < K

        # 6a. Load A Tile
        # Load `BLOCK_SIZE_M x BLOCK_SIZE_K` tile from A
        # Apply masking for K dimension boundaries, load 0 for out-of-bounds
        a = tl.load(A_tile_ptr, mask=k_boundary_mask_a, other=0.0)

        # 6b. Load B Tile
        # Load `BLOCK_SIZE_K x BLOCK_SIZE_N` tile from B
        # Apply masking for K dimension boundaries, load 0 for out-of-bounds
        b = tl.load(B_tile_ptr, mask=k_boundary_mask_b, other=0.0)

        # 6c. Matrix Multiply Tile
        # Compute the dot product of the tiles
        # Allow TF32 for potentially faster computation on compatible hardware
        accumulator += tl.dot(a, b, allow_tf32=True)

        # 6e. Update Pointers
        # Advance pointers to the next K block
        A_tile_ptr += BLOCK_SIZE_K * stride_ak
        B_tile_ptr += BLOCK_SIZE_K * stride_bk

    # 7. Cast Accumulator
    # Cast the float32 accumulator to the output data type (e.g., float16)
    c = accumulator.to(C_ptr.dtype.element_ty)

    # 8. Calculate C Pointers
    # Define pointers for the output C tile: starts at row `offs_m`, column `offs_n`
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # 9. Store Result
    # Define boundary masks for M and N dimensions
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Write the computed `BLOCK_SIZE_M x BLOCK_SIZE_N` tile to C
    # Apply masking for M and N boundaries to avoid out-of-bounds writes
    tl.store(C_tile_ptr, c, mask=output_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Performs matrix multiplication C = A @ B using the Triton kernel `_matmul_kernel`.
        It handles the setup and kernel launch.

    Args:
        A (torch.Tensor): The first input matrix. [Shape: (M, K)]
        B (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The result matrix C. [Shape: (M, N)]

    Explanation:
        1. Input Validation: Check if the input tensors A and B are 2D tensors.
        2. Dimension Extraction: Get the dimensions M, K from A and K, N from B. Validate that the inner dimension K matches.
        3. Output Tensor Creation: Create an empty tensor C with the shape (M, N) on the same device and with the same data type as the input tensors (or promote types if necessary, here using A's dtype).
        4. Grid Calculation: Define the grid for launching the Triton kernel. The grid size determines the number of kernel instances (programs). It's calculated based on the number of blocks needed to cover the M and N dimensions, potentially adjusted by GROUP_SIZE_M implicitly through the kernel's logic. The grid is typically `(num_m_blocks * num_n_blocks, 1, 1)`.
        5. Kernel Launch: Call the `_matmul_kernel` with the calculated grid. Pass the input tensors (A, B), output tensor (C), matrix dimensions (M, N, K), and strides. Triton automatically extracts pointers and strides from the tensors. The autotuner selects the best `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, and `GROUP_SIZE_M` based on the provided configurations and input dimensions.
        6. Return Result: Return the output tensor C containing the result of the matrix multiplication.
    """
    # 1. Input Validation (Basic)
    assert A.dim() == 2, "Matrix A must be 2-dimensional"
    assert B.dim() == 2, "Matrix B must be 2-dimensional"
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA device"
    # Assuming A and B have compatible data types or promotion is acceptable.
    # Using A's dtype and device for the output.
    output_dtype = A.dtype
    device = A.device

    # 2. Dimension Extraction
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, f"Inner dimensions must match: A.shape[1]={K}, B.shape[0]={K_B}"

    # 3. Output Tensor Creation
    # Create output tensor C on the same device and dtype as A
    C = torch.empty((M, N), device=device, dtype=output_dtype)

    # 4. Grid Calculation
    # The grid determines the number of program instances.
    # Each program computes a block of the output matrix.
    # The autotuner will select block sizes, but we define the grid logic here.
    # The kernel internally handles mapping the 1D grid ID to 2D blocks + grouping.
    def grid(meta):
        # Calculate the number of blocks needed based on M and N dimensions
        # These block sizes are placeholders here; the autotuner provides the actual ones.
        # The kernel uses tl.cdiv for calculation, so we do too.
        num_m_blocks = tl.cdiv(M, meta['BLOCK_SIZE_M'])
        num_n_blocks = tl.cdiv(N, meta['BLOCK_SIZE_N'])
        # The kernel expects a 1D grid where pid is calculated from this total number.
        return (num_m_blocks * num_n_blocks, ) # Triton expects a tuple

    # 5. Kernel Launch
    # Call the kernel, passing tensors (pointers/strides extracted automatically),
    # dimensions, and let autotuner handle BLOCK_* and GROUP_SIZE_*.
    _matmul_kernel[grid](
        A, B, C,                    # Pass tensors directly
        M, N, K,                    # Pass dimensions
        A.stride(0), A.stride(1),   # Strides for A
        B.stride(0), B.stride(1),   # Strides for B
        C.stride(0), C.stride(1)    # Strides for C
        # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M are implicitly passed by autotuner
    )

    # 6. Return Result
    return C

def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Computes matrix multiplication C = A @ B using PyTorch's built-in `torch.matmul`.
        This serves as a reference implementation for correctness and performance comparison.

    Args:
        A (torch.Tensor): The first input matrix. [Shape: (M, K)]
        B (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The result matrix C. [Shape: (M, N)]

    Explanation:
        1. Call `torch.matmul`: Directly use the PyTorch function `torch.matmul(A, B)` to perform the matrix multiplication.
        2. Return Result: Return the tensor computed by PyTorch.
    """
    # 1. Call torch.matmul
    C = torch.matmul(A, B)
    # 2. Return Result
    return C


# Main execution block for benchmarking
if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        # A = torch.nn.functional.normalize(A, dim=-1) # Normalization can affect correctness checks at higher tolerances
        B = torch.randn(K, N, device=device, dtype=dtype)
        # B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func

    # Use higher tolerance for float16 due to potential precision differences and TF32 usage
    # Correctness check (Triton vs PyTorch)
    print("Running correctness check...")
    verify_correctness_func(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes], atol=0.5, rtol=0.1)

    # Performance benchmark (Triton vs PyTorch [compiled])
    print("\nRunning performance benchmark...")
    benchmark_performance_func(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])

