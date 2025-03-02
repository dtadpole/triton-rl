# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Performs matrix multiplication using Triton with autotuning capabilities.

    Args:
        a_ptr (triton.Pointer): Pointer to matrix A. Shape (M, K)
        b_ptr (triton.Pointer): Pointer to matrix B. Shape (K, N)
        c_ptr (triton.Pointer): Pointer to output matrix C. Shape (M, N)
        M (int): Number of rows in matrix A and C
        N (int): Number of columns in matrix B and C
        K (int): Number of columns in matrix A and rows in matrix B
        stride_am (int): Stride between rows of matrix A
        stride_ak (int): Stride between columns of matrix A
        stride_bk (int): Stride between rows of matrix B
        stride_bn (int): Stride between columns of matrix B
        stride_cm (int): Stride between rows of matrix C
        stride_cn (int): Stride between columns of matrix C
        BLOCK_SIZE_M (tl.constexpr): Block size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Block size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Block size for K dimension
        GROUP_SIZE_M (tl.constexpr): Group size for M dimension to improve L2 cache hit rate

    Returns:
        None: Output is stored in c_ptr

    Explanation:
        1. Calculate program ID and corresponding block indices
        2. Create pointers for blocks in A and B matrices
        3. Initialize accumulator block with zeros
        4. Perform block-wise matrix multiplication with double buffering
        5. Store results in output matrix C
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using Triton kernel with autotuning.

    Args:
        a (torch.Tensor): Input matrix A. Shape (M, K)
        b (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)

    Explanation:
        1. Validate input dimensions
        2. Preallocate output tensor
        3. Calculate grid dimensions based on autotuned block sizes
        4. Launch optimized kernel with autotuned parameters
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda opt: (triton.cdiv(M, opt['BLOCK_SIZE_M']) * triton.cdiv(N, opt['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        GROUP_SIZE_M=8
    )
    return c

def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs standard matrix multiplication using PyTorch.

    Args:
        a (torch.Tensor): Input matrix A. Shape (M, K)
        b (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)

    Explanation:
        1. Directly uses torch.matmul for computation
        2. Serves as reference implementation for verification
    """
    return torch.matmul(a, b)

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
