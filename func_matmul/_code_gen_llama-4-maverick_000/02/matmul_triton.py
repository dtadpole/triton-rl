# filename: matmul_triton.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Description: Triton kernel for matrix multiplication.

    Args:
        A_ptr (triton.Pointer): Pointer to matrix A. [M, K]
        B_ptr (triton.Pointer): Pointer to matrix B. [K, N]
        C_ptr (triton.Pointer): Pointer to output matrix C. [M, N]
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride of A along dimension M.
        stride_ak (int): Stride of A along dimension K.
        stride_bk (int): Stride of B along dimension K.
        stride_bn (int): Stride of B along dimension N.
        stride_cm (int): Stride of C along dimension M.
        stride_cn (int): Stride of C along dimension N.
        BLOCK_SIZE_M (int): Block size for dimension M.
        BLOCK_SIZE_N (int): Block size for dimension N.
        BLOCK_SIZE_K (int): Block size for dimension K.
        GROUP_SIZE_M (int): Group size for dimension M.

    Returns:
        None

    Explanation:
        1. Calculate the row and column indices for the output matrix C based on the thread's position.
        2. Initialize accumulators for the dot product.
        3. Load data from A and B into shared memory in blocks.
        4. Perform the dot product using the loaded data.
        5. Store the result in the output matrix C.
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

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator)

def matmul_triton(A, B):
    """
    Description: Python wrapper for Triton matrix multiplication kernel.

    Args:
        A (torch.Tensor): Input matrix A. [M, K]
        B (torch.Tensor): Input matrix B. [K, N]

    Returns:
        torch.Tensor: Result of matrix multiplication. [M, N]

    Explanation:
        1. Get the shapes of input matrices A and B.
        2. Allocate output matrix C.
        3. Call the Triton kernel for matrix multiplication.
        4. Return the result.
    """
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C

def generated_torch_func(A, B):
    """
    Description: PyTorch implementation of matrix multiplication for comparison.

    Args:
        A (torch.Tensor): Input matrix A. [M, K]
        B (torch.Tensor): Input matrix B. [K, N]

    Returns:
        torch.Tensor: Result of matrix multiplication. [M, N]
    """
    return torch.matmul(A, B)

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
    verify_correctness_func(matmul_triton, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(matmul_triton, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
