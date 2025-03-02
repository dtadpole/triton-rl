# filename: matmul_triton.py
import torch
import triton
import triton.language as tl
from triton.ops.matmul import get_gpu_autotune_config

def torch_matmul(A, B):
    return torch.matmul(A, B)

@triton.autotune(
    configs=get_gpu_autotune_config('matmul'),
    key=['M', 'N', 'K'],
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
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Performs matrix multiplication using Triton with autotuning for block sizes.

    Args:
        A_ptr (triton.Pointer): Pointer to matrix A of shape (M, K).
        B_ptr (triton.Pointer): Pointer to matrix B of shape (K, N).
        C_ptr (triton.Pointer): Pointer to output matrix C of shape (M, N).
        M (int): Number of rows in matrix A.
        N (int): Number of columns in matrix B.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride between rows in matrix A.
        stride_ak (int): Stride between columns in matrix A.
        stride_bk (int): Stride between rows in matrix B.
        stride_bn (int): Stride between columns in matrix B.
        stride_cm (int): Stride between rows in output matrix C.
        stride_cn (int): Stride between columns in output matrix C.
        BLOCK_SIZE_M (tl.constexpr): Block size for M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for K dimension.
        GROUP_SIZE_M (tl.constexpr): Group size for M dimension in super-blocking.

    Returns:
        None: Output is stored in C_ptr.

    Explanation:
        1. Calculate program ID for current operation.
        2. Form super-blocks to optimize L2 cache usage.
        3. Load blocks of matrices A and B into SRAM.
        4. Compute matrix multiplication using accumulated dot product.
        5. Write back results to output matrix C.
    """
    pid = tl.program_id(0)
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
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c)

def matmul_triton(A, B):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda opt: (
        triton.cdiv(M, opt['BLOCK_SIZE_M']) * triton.cdiv(N, opt['BLOCK_SIZE_N']),
        1,
    )
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        GROUP_SIZE_M=8
    )
    return C

M_sizes = [512, 1024, 2048, 4096]
N_sizes = [512, 1024, 2048, 4096]
K_sizes = [512, 1024, 2048, 4096]

def input_generator(M, N, K, device='cuda', dtype=torch.float16):
    A = torch.randn(M, K, device=device, dtype=dtype)
    A = torch.nn.functional.normalize(A, dim=-1)
    B = torch.randn(K, N, device=device, dtype=dtype)
    B = torch.nn.functional.normalize(B, dim=-1)
    return A, B

from benchmark import verify_correctness, benchmark_performance
verify_correctness(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
benchmark_performance(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
