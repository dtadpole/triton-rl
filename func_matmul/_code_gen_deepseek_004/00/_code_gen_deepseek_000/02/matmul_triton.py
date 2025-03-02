# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    TRITON kernel for matrix multiplication with autotuned block configurations.

    Args:
        A_ptr (triton.Pointer): Pointer to matrix A. Shape (M, K)
        B_ptr (triton.Pointer): Pointer to matrix B. Shape (K, N)
        C_ptr (triton.Pointer): Pointer to output matrix C. Shape (M, N)
        M (int): Number of rows in matrix A and C
        N (int): Number of columns in matrix B and C
        K (int): Number of columns in A and rows in B
        stride_am (int): Stride along M dimension for A
        stride_ak (int): Stride along K dimension for A
        stride_bk (int): Stride along K dimension for B
        stride_bn (int): Stride along N dimension for B
        stride_cm (int): Stride along M dimension for C
        stride_cn (int): Stride along N dimension for C
        BLOCK_SIZE_M (tl.constexpr): Autotuned block size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Autotuned block size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Autotuned block size for K dimension

    Explanation:
        1. Calculate block IDs for program grid
        2. Initialize pointers for input matrices A and B
        3. Iterate through K dimension using block-level matrix multiplication
        4. Accumulate results in local register
        5. Write final results to output matrix C
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_mask = (offs_k[:, None] < K - k) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(C_ptr.dtype.element_ty), mask=c_mask)

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    TRITON-accelerated matrix multiplication with autotuning.

    Args:
        A (torch.Tensor): Input matrix A. Shape (M, K)
        B (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)

    Explanation:
        1. Validate input dimensions
        2. Preallocate output tensor
        3. Calculate grid dimensions
        4. Launch autotuned kernel with appropriate parameters
    """
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda opt: (triton.cdiv(M, opt.BLOCK_SIZE_M) * triton.cdiv(N, opt.BLOCK_SIZE_N),)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=64,  # Default values for autotune parameters
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    return C

def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of matrix multiplication for verification.

    Args:
        A (torch.Tensor): Input matrix A. Shape (M, K)
        B (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)
    """
    return torch.matmul(A, B)

if __name__ == "__main__":
    M_sizes = [256, 512, 1024, 2048, 4096]
    N_sizes = [256, 512, 1024, 2048, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B
    
    verify_correctness(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
