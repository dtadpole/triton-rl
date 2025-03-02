# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
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
    Description: Performs matrix multiplication using Triton with autotuned block configurations.

    Args:
        a_ptr (tl.pointer): Pointer to input matrix A of shape (M, K)
        b_ptr (tl.pointer): Pointer to input matrix B of shape (K, N)
        c_ptr (tl.pointer): Pointer to output matrix C of shape (M, N)
        M (int): Number of rows in matrix A and C
        N (int): Number of columns in matrix B and C
        K (int): Number of columns in matrix A and rows in matrix B
        stride_am (int): Stride between rows of A
        stride_ak (int): Stride between columns of A
        stride_bk (int): Stride between rows of B
        stride_bn (int): Stride between columns of B
        stride_cm (int): Stride between rows of C
        stride_cn (int): Stride between columns of C
        BLOCK_SIZE_M (tl.constexpr): Autotuned block size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Autotuned block size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Autotuned block size for K dimension
        GROUP_SIZE_M (tl.constexpr): Autotuned group size for scheduling

    Returns:
        None: Output is written through c_ptr

    Explanation:
        1. Calculate program ID and group ID for workload distribution
        2. Determine block boundaries for M and N dimensions
        3. Initialize accumulator with zeros
        4. Iterate over K dimension in blocks, loading and accumulating values
        5. Store final results to output matrix C
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

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Description: Wrapper function for Triton matrix multiplication kernel.

    Args:
        a (torch.Tensor): Input matrix A of shape (M, K)
        b (torch.Tensor): Input matrix B of shape (K, N)

    Returns:
        torch.Tensor: Output matrix C of shape (M, N)

    Explanation:
        1. Verify input dimensions match matrix multiplication requirements
        2. Pre-allocate output tensor
        3. Calculate grid dimensions based on block sizes
        4. Launch Triton kernel with appropriate parameters
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    M_sizes = [512, 1024, 2048, 4096]
    N_sizes = [512, 1024, 2048, 4096]
    K_sizes = [512, 1024, 2048, 4096]

    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    verify_correctness(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
