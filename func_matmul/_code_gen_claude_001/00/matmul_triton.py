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
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how to access the next element in a row/column
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Matrix multiplication kernel for computing C = A @ B.

    Args:
        a_ptr (ptr): Pointer to the A matrix. [M, K]
        b_ptr (ptr): Pointer to the B matrix. [K, N]
        c_ptr (ptr): Pointer to the output C matrix. [M, N]
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride for moving along the rows of A.
        stride_ak (int): Stride for moving along the columns of A.
        stride_bk (int): Stride for moving along the rows of B.
        stride_bn (int): Stride for moving along the columns of B.
        stride_cm (int): Stride for moving along the rows of C.
        stride_cn (int): Stride for moving along the columns of C.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.
        GROUP_SIZE_M (tl.constexpr): Number of blocks to group in the M dimension.

    Returns:
        None: Results are written to c_ptr.

    Explanation:
        1. Calculate program ID for the blocks of output matrix C to compute.
        2. Compute row and column indices for the current thread block.
        3. Create pointers to the relevant blocks of A, B, and C.
        4. Iterate through the K dimension in blocks, loading data and performing matrix multiplication.
        5. Accumulate results in a local variable and finally write them to global memory.
    """
    # --- Super-blocking ---
    # Programs are grouped in the M dimension to improve L2 hit rate
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Set boundary conditions and block memory addresses
    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Boundary check to prevent out-of-bounds accesses
    m_mask = m_offs < M
    n_mask = n_offs < N

    # Make the offsets of A and B at this block
    a_offs = m_offs[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak
    b_offs = tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + n_offs[None, :] * stride_bn

    # Accumulator for results
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate through the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_size = min(k_remaining, BLOCK_SIZE_K)
        
        # Create a mask for the current K block
        k_mask = tl.arange(0, BLOCK_SIZE_K) < k_size

        # Load the inputs and mask out data outside the K boundary
        a = tl.load(a_ptr + a_offs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Compute matrix multiplication for the current block
        acc += tl.dot(a, b)
        
        # Move to the next K block
        a_offs += BLOCK_SIZE_K * stride_ak
        b_offs += BLOCK_SIZE_K * stride_bk

    # Create pointers to C matrix
    c_offs = m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    
    # Write accumulated results back to C
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offs, acc, mask=c_mask)

def generated_triton_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton matrix multiplication kernel.

    Args:
        a (torch.Tensor): Input matrix A of shape [M, K].
        b (torch.Tensor): Input matrix B of shape [K, N].

    Returns:
        torch.Tensor: Result matrix C of shape [M, N].

    Explanation:
        1. Extract dimensions from input matrices.
        2. Create an output tensor to store the results.
        3. Calculate the strides for matrices A, B, and C.
        4. Compute the grid dimensions for the kernel launch.
        5. Launch the Triton kernel to compute matrix multiplication.
        6. Return the computed output matrix.
    """
    # Extract dimensions from input matrices
    M, K = a.shape
    K_, N = b.shape
    
    # Verify that the common dimension matches
    assert K == K_, "Incompatible dimensions for matrix multiplication"
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Calculate strides for matrices
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)
    
    # Compute grid dimensions: number of blocks needed
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    
    return c

def generated_torch_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of matrix multiplication.

    Args:
        a (torch.Tensor): Input matrix A of shape [M, K].
        b (torch.Tensor): Input matrix B of shape [K, N].

    Returns:
        torch.Tensor: Result matrix C of shape [M, N].

    Explanation:
        1. Uses PyTorch's native matrix multiplication operation.
        2. Returns the result of A @ B.
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
