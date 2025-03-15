# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
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
):
    """
    Compute matrix multiplication C = A @ B using tiled approach with super blocks.

    Args:
        a_ptr (tl.pointer): Pointer to matrix A. [M, K]
        b_ptr (tl.pointer): Pointer to matrix B. [K, N]
        c_ptr (tl.pointer): Pointer to output matrix C. [M, N]
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride for moving down one row in A.
        stride_ak (int): Stride for moving right one column in A.
        stride_bk (int): Stride for moving down one row in B.
        stride_bn (int): Stride for moving right one column in B.
        stride_cm (int): Stride for moving down one row in C.
        stride_cn (int): Stride for moving right one column in C.
        BLOCK_SIZE_M (tl.constexpr): Size of tile in M dimension.
        BLOCK_SIZE_N (tl.constexpr): Size of tile in N dimension.
        BLOCK_SIZE_K (tl.constexpr): Size of tile in K dimension.
        GROUP_SIZE_M (tl.constexpr): Number of M-dimension blocks per group.

    Returns:
        None: Results are written to c_ptr.

    Explanation:
        1. Compute block grid positions based on program ID with super block grouping.
        2. Create block pointers into A, B, and C matrices.
        3. Initialize the accumulator for result tile.
        4. Iterate through K dimension in tiles, loading parts of A and B.
        5. Perform matrix multiplication on the loaded tiles and accumulate.
        6. Write the final accumulated result to matrix C.
    """
    # -----------------------------------------------------------
    # Map program ID to the blocks of C that each program will compute
    # -----------------------------------------------------------
    # Program ID
    pid = tl.program_id(0)
    
    # Number of blocks in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Number of programs in a group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID and local ID within the group
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Local position within the group
    local_id = pid % num_pid_in_group
    pid_m = first_pid_m + (local_id % group_size_m)
    pid_n = local_id // group_size_m

    # -----------------------------------------------------------
    # Create block pointers for the current block
    # -----------------------------------------------------------
    # M and N offsets for the current block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers to blocks in matrices A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # -----------------------------------------------------------
    # Compute the matrix multiplication for this block
    # -----------------------------------------------------------
    # Initialize accumulator for the result
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate through the K dimension in tiles
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Check bounds for the current K tile
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Create masks for bounds checking
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        # Load blocks of A and B, accounting for bounds
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Convert to float32 for higher precision in accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        
        # Perform matrix multiplication for this K tile
        accumulator += tl.dot(a, b)
        
        # Advance the pointers to the next K tile
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # -----------------------------------------------------------
    # Write the result to matrix C
    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create pointers to the output matrix C
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Create mask for bounds checking
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Store the result
    tl.store(c_ptrs, accumulator, mask=c_mask)

def generated_triton_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using Triton kernel.

    Args:
        a (torch.Tensor): Input matrix A of shape [M, K].
        b (torch.Tensor): Input matrix B of shape [K, N].

    Returns:
        torch.Tensor: Result matrix C of shape [M, N].

    Explanation:
        1. Extract dimensions from input matrices.
        2. Verify compatibility of inner dimensions for matrix multiplication.
        3. Allocate memory for the output matrix C.
        4. Compute the grid size based on matrix dimensions.
        5. Launch the Triton kernel to perform the matrix multiplication.
        6. Return the computed result matrix.
    """
    # Extract matrix dimensions
    M, K = a.shape
    K_b, N = b.shape
    
    # Verify that matrices can be multiplied
    assert K == K_b, f"Incompatible dimensions for matrix multiplication: {a.shape} and {b.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Compute grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c

def generated_torch_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using PyTorch's built-in function.

    Args:
        a (torch.Tensor): Input matrix A of shape [M, K].
        b (torch.Tensor): Input matrix B of shape [K, N].

    Returns:
        torch.Tensor: Result matrix C of shape [M, N].

    Explanation:
        1. Use PyTorch's built-in matmul function to multiply matrices A and B.
        2. Return the resulting matrix.
    """
    return torch.matmul(a, b)

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
