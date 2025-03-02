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
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
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
):
    """
    Triton kernel for matrix multiplication with autotuning support.

    Args:
        a_ptr (tl.pointer): Pointer to input matrix A. Shape (M, K)
        b_ptr (tl.pointer): Pointer to input matrix B. Shape (K, N)
        c_ptr (tl.pointer): Pointer to output matrix C. Shape (M, N)
        M (int): Number of rows in matrix A
        N (int): Number of columns in matrix B
        K (int): Number of columns in matrix A / rows in matrix B
        stride_am (int): Stride along M dimension for matrix A
        stride_ak (int): Stride along K dimension for matrix A
        stride_bk (int): Stride along K dimension for matrix B
        stride_bn (int): Stride along N dimension for matrix B
        stride_cm (int): Stride along M dimension for matrix C
        stride_cn (int): Stride along N dimension for matrix C
        BLOCK_SIZE_M (tl.constexpr): Block size along M dimension
        BLOCK_SIZE_N (tl.constexpr): Block size along N dimension
        BLOCK_SIZE_K (tl.constexpr): Block size along K dimension

    Returns:
        None: Output is stored through c_ptr

    Explanation:
        1. Calculate program ID and super block offsets
        2. Initialize pointers for input matrices A and B
        3. Initialize accumulator with zeros
        4. Perform block matrix multiplication through accumulation
        5. Write back results to output matrix C
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

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
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using Triton kernel.

    Args:
        a (torch.Tensor): Input matrix A. Shape (M, K)
        b (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)

    Explanation:
        1. Validate input dimensions
        2. Preallocate output tensor
        3. Calculate grid size based on input dimensions
        4. Launch Triton kernel with autotuned parameters
    """
    assert a.shape[1] == b.shape[0], "Incompatible matrix dimensions"
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

def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using PyTorch's built-in function.

    Args:
        a (torch.Tensor): Input matrix A. Shape (M, K)
        b (torch.Tensor): Input matrix B. Shape (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape (M, N)

    Explanation:
        1. Directly use torch.matmul for matrix multiplication
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
    
    from benchmark import verify_correctness, benchmark_performance
    verify_correctness(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
