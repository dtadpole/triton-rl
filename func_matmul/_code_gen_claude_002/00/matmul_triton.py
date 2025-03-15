# filename: matmul_triton.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access the next element
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing the matrix multiplication C = A @ B.

    Args:
        a_ptr (tl.pointer): Pointer to the A matrix. [M x K]
        b_ptr (tl.pointer): Pointer to the B matrix. [K x N]
        c_ptr (tl.pointer): Pointer to the C matrix for storing results. [M x N]
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride for moving along rows in A.
        stride_ak (int): Stride for moving along columns in A.
        stride_bk (int): Stride for moving along rows in B.
        stride_bn (int): Stride for moving along columns in B.
        stride_cm (int): Stride for moving along rows in C.
        stride_cn (int): Stride for moving along columns in C.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_M (tl.constexpr): Group size for the M dimension when using super blocks.

    Returns:
        None: Results are written to c_ptr.

    Explanation:
        1. Map program ID to the blocks of C being computed using super blocks.
        2. Create block pointers for A, B, and C matrices.
        3. Initialize accumulator to zeros.
        4. Loop over K dimension in blocks.
        5. Load blocks from A and B, handling boundary conditions.
        6. Perform matrix multiplication for the current block.
        7. Store the accumulated result in C.
    """
    # Map program ID to the blocks of C being computed
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Using super blocks: reorganize the programs into groups along the M dimension
    # Each group of programs handles GROUP_SIZE_M blocks of M
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Calculate the program's position within the group
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Calculate offsets for accessing A, B, and C matrices
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create pointers to blocks in A and B matrices
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # Initialize accumulator for matrix multiplication
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate remaining elements in K dimension
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Create masks for boundary conditions
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        # Load blocks from A and B matrices, with boundary checks
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Convert to float32 for higher precision in accumulation
        a = tl.float32(a)
        b = tl.float32(b)
        
        # Perform matrix multiplication for current block
        accumulator += tl.dot(a, b)
        
        # Move to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Calculate pointers to output C matrix
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    
    # Create mask for output (handles boundaries)
    mask_c = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    
    # Store the result in C matrix
    tl.store(c_ptrs, accumulator, mask=mask_c)

def triton_matmul(a, b):
    """
    Perform matrix multiplication using Triton: C = A @ B
    
    Args:
        a (torch.Tensor): Input tensor of shape (M, K). [M x K]
        b (torch.Tensor): Input tensor of shape (K, N). [K x N]
    
    Returns:
        torch.Tensor: Result tensor of shape (M, N). [M x N]
    
    Explanation:
        1. Extract dimensions from input matrices.
        2. Verify that the dimensions are compatible for matrix multiplication.
        3. Create the output matrix to store results.
        4. Calculate grid size based on the matrix dimensions and block size.
        5. Launch the Triton kernel with appropriate parameters.
        6. Return the computed result.
    """
    # Extract dimensions from input matrices
    M, K = a.shape
    Kb, N = b.shape
    
    # Verify matrices have compatible dimensions
    assert K == Kb, f"Incompatible dimensions for matrix multiplication: {a.shape} and {b.shape}"
    
    # Create output matrix
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define grid - total number of programs to launch
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

def torch_matmul(a, b):
    """
    Perform matrix multiplication using PyTorch: C = A @ B
    
    Args:
        a (torch.Tensor): Input tensor of shape (M, K). [M x K]
        b (torch.Tensor): Input tensor of shape (K, N). [K x N]
    
    Returns:
        torch.Tensor: Result tensor of shape (M, N). [M x N]
    
    Explanation:
        1. Use PyTorch's built-in matmul operation to multiply matrices A and B.
        2. Return the computed result.
    """
    return torch.matmul(a, b)

# For convenience, create aliases for the generated functions
generated_triton_func = triton_matmul
generated_torch_func = torch_matmul

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
