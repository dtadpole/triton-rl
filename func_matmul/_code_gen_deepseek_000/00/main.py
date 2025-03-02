# filename: main.py
import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Performs matrix multiplication using Triton with tiling and super-block optimization.

    Args:
        a_ptr (tl.pointer): Pointer to input matrix A. Shape: (M, K)
        b_ptr (tl.pointer): Pointer to input matrix B. Shape: (K, N)
        c_ptr (tl.pointer): Pointer to output matrix C. Shape: (M, N)
        M (int): Number of rows in matrix A
        N (int): Number of columns in matrix B
        K (int): Number of columns in matrix A / rows in matrix B
        stride_am (int): Stride along M dimension of A
        stride_ak (int): Stride along K dimension of A
        stride_bk (int): Stride along K dimension of B
        stride_bn (int): Stride along N dimension of B
        stride_cm (int): Stride along M dimension of C
        stride_cn (int): Stride along N dimension of C
        BLOCK_SIZE_M (tl.constexpr): Tiling size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Tiling size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Tiling size for K dimension
        GROUP_SIZE_M (tl.constexpr): Super-block grouping size for M dimension

    Returns:
        None: Output is stored through c_ptr

    Explanation:
        1. Calculate program IDs with super-block grouping
        2. Initialize accumulator with zeros
        3. Load tiles from A and B matrices
        4. Compute matrix multiplication with double buffering
        5. Store final accumulator to output matrix
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
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton matrix multiplication wrapper function.

    Args:
        a (torch.Tensor): Input matrix A. Shape: (M, K)
        b (torch.Tensor): Input matrix B. Shape: (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape: (M, N)

    Explanation:
        1. Preallocate output tensor
        2. Calculate grid size based on input dimensions
        3. Launch optimized Triton kernel
    """
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda opt: (triton.cdiv(M, opt['BLOCK_SIZE_M']) * triton.cdiv(N, opt['BLOCK_SIZE_N']),)
    matmul_kernel[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1),
                       b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1))
    return c

def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch matrix multiplication reference implementation.

    Args:
        a (torch.Tensor): Input matrix A. Shape: (M, K)
        b (torch.Tensor): Input matrix B. Shape: (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape: (M, N)

    Explanation:
        1. Perform standard matrix multiplication using torch.mm
    """
    return torch.mm(a, b)

if __name__ == "__main__":
    M_sizes = [512, 1024, 2048, 4096]
    N_sizes = [512, 1024, 2048, 4096]
    K_sizes = [512, 1024, 2048, 4096]

    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=0)
        return A, B

    verify_correctness(triton_matmul, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(triton_matmul, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
