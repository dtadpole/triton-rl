# filename: matmul_triton.py
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing C = A @ B where A has shape (M, K) and B has shape (K, N).

    Args:
        a_ptr (pointer): Pointer to the A matrix [M, K].
        b_ptr (pointer): Pointer to the B matrix [K, N].
        c_ptr (pointer): Pointer to the output C matrix [M, N].
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride for moving one row in A.
        stride_ak (int): Stride for moving one column in A.
        stride_bk (int): Stride for moving one row in B.
        stride_bn (int): Stride for moving one column in B.
        stride_cm (int): Stride for moving one row in C.
        stride_cn (int): Stride for moving one column in C.
        BLOCK_SIZE_M (tl.constexpr): Size of the tile along M dimension.
        BLOCK_SIZE_N (tl.constexpr): Size of the tile along N dimension.
        BLOCK_SIZE_K (tl.constexpr): Size of the tile along K dimension.
        GROUP_SIZE_M (tl.constexpr): Number of blocks to group together along M dimension.

    Explanation:
        1. Calculate grid dimensions and program ID to determine which block to process.
        2. Use super blocks with GROUP_SIZE_M to enhance cache locality.
        3. Calculate offsets for the current block in matrices A, B, and C.
        4. Initialize an accumulator for the result.
        5. Iterate through K dimension in tiles of size BLOCK_SIZE_K.
        6. Load blocks of A and B matrices with boundary check.
        7. Perform matrix multiplication (dot product) of the loaded blocks.
        8. Accumulate results to the output matrix.
        9. Write the final result to memory with boundary check.
    """
    # Compute number of blocks in each dimension
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Compute program ID in group-based format
    pid = tl.program_id(axis=0)
    
    # Number of programs in a group
    num_programs_in_group = GROUP_SIZE_M * grid_n
    
    # Group ID
    group_id = pid // num_programs_in_group
    
    # Local ID within the group
    local_id = pid % num_programs_in_group
    
    # Row and column IDs
    local_m = local_id // grid_n
    n = local_id % grid_n
    
    # Global row ID
    m = group_id * GROUP_SIZE_M + local_m
    
    # Check if we're out of bounds
    if m >= grid_m:
        return
    
    # Block starting indices
    block_m = m * BLOCK_SIZE_M
    block_n = n * BLOCK_SIZE_N
    
    # Offsets for each thread within the block
    offs_m = block_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Current K block offset
        k_offset = k * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        
        # Create masks for boundary conditions
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # Compute pointers for current blocks of A and B
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        # Load data with boundary check
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Compute matrix multiplication for this block
        accumulator += tl.dot(a, b)
    
    # Compute pointers for C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    
    # Create mask for C to handle boundary conditions
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Store the result
    tl.store(c_ptrs, accumulator, mask=mask_c)


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using Triton.

    Args:
        a (torch.Tensor): Input matrix of shape [M, K].
        b (torch.Tensor): Input matrix of shape [K, N].

    Returns:
        torch.Tensor: Result matrix of shape [M, N].

    Explanation:
        1. Verify matrix dimensions are compatible (K dimension matches).
        2. Allocate output tensor of appropriate shape.
        3. Calculate grid size based on matrix dimensions and block sizes.
        4. Launch the Triton kernel to perform the multiplication.
        5. Return the resulting matrix.
    """
    # Get the shapes
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions for matrix multiplication: {a.shape} and {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Calculate the grid size based on block sizes and grouping
    def grid(META):
        grid_m = triton.cdiv(M, META['BLOCK_SIZE_M'])
        grid_n = triton.cdiv(N, META['BLOCK_SIZE_N'])
        # Number of groups needed (ceiling division)
        num_groups = (grid_m + META['GROUP_SIZE_M'] - 1) // META['GROUP_SIZE_M']
        # Total number of programs to launch
        return (num_groups * META['GROUP_SIZE_M'] * grid_n,)
    
    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c


def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using PyTorch.

    Args:
        a (torch.Tensor): Input matrix of shape [M, K].
        b (torch.Tensor): Input matrix of shape [K, N].

    Returns:
        torch.Tensor: Result matrix of shape [M, N].

    Explanation:
        1. Uses PyTorch's built-in matrix multiplication.
        2. This function serves as a reference implementation to compare with the Triton version.
    """
    return torch.matmul(a, b)


# For testing
generated_triton_func = matmul_triton
generated_torch_func = matmul_torch

if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
