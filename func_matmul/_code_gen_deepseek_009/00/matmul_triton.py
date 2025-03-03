# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Computes matrix multiplication C = A x B using Triton with autotuned blocking.

    Args:
        a_ptr (tl.tensor): Pointer to input matrix A. Shape (M, K)
        b_ptr (tl.tensor): Pointer to input matrix B. Shape (K, N)
        c_ptr (tl.tensor): Pointer to output matrix C. Shape (M, N)
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
        GROUP_SIZE_M (tl.constexpr): Autotuned group size for super-blocking

    Returns:
        None: Output is stored through c_ptr

    Explanation:
        1. Calculate program ID and group ID for super-block partitioning
        2. Compute block ranges for M and N dimensions using autotuned parameters
        3. Initialize accumulator with zeros
        4. Iterate over K dimension in blocks, loading A and B tiles
        5. Perform matrix multiplication on current tile pair
        6. Store accumulated result to output matrix C
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

    block_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (block_m[:, None] * stride_am + block_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (block_k[:, None] * stride_bk + block_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(block_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(block_k[:, None] < K - k), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (block_m[:, None] * stride_cm + block_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    TRITON-accelerated matrix multiplication with autotuned blocking.

    Args:
        a (torch.Tensor): Input matrix A. Shape (M, K)
        b (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)

    Explanation:
        1. Validate input dimensions for matrix multiplication
        2. Preallocate output matrix C
        3. Calculate grid size based on autotuned block parameters
        4. Launch optimized Triton kernel with calculated grid
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda opt: (triton.cdiv(M, opt.BLOCK_SIZE_M) * triton.cdiv(N, opt.BLOCK_SIZE_N),)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Baseline PyTorch matrix multiplication"""
    return torch.matmul(a, b)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance
    
    M_sizes = [256, 512, 1024, 2048, 4096]
    N_sizes = [256, 512, 1024, 2048, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B
    
    verify_correctness(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
