# filename: matmul_triton.py
import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
            num_warps=8
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
            num_warps=8
        ),
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
    GROUP_SIZE_M: tl.constexpr
):
    """
    Computes matrix multiplication C = A @ B using Triton with autotuning.
    
    Args:
        a_ptr (triton.Pointer): Pointer to matrix A of shape (M, K)
        b_ptr (triton.Pointer): Pointer to matrix B of shape (K, N)
        c_ptr (triton.Pointer): Pointer to output matrix C of shape (M, N)
        M (int): Number of rows in A and C
        N (int): Number of columns in B and C
        K (int): Number of columns in A and rows in B
        stride_am (int): Stride of A along M dimension
        stride_ak (int): Stride of A along K dimension
        stride_bk (int): Stride of B along K dimension
        stride_bn (int): Stride of B along N dimension
        stride_cm (int): Stride of C along M dimension
        stride_cn (int): Stride of C along N dimension
        BLOCK_SIZE_M (tl.constexpr): Tile size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Tile size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Tile size for K dimension
        GROUP_SIZE_M (tl.constexpr): Number of M blocks to group together
    
    Returns:
        None: Output is stored in c_ptr
    
    Explanation:
        1. Calculate program ID and corresponding block indices for C matrix
        2. Create block pointers for A and B matrices with proper offsets
        3. Initialize accumulator for C block with zeros
        4. Iterate over K dimension in chunks of BLOCK_SIZE_K
        5. Load blocks from A and B matrices, cast to float32 for computation
        6. Accumulate matrix product into C block using tl.dot
        7. Write final accumulated result back to C matrix
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
    
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1)
    )
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(1, 0))
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator.to(tl.float16), boundary_check=(0, 1))

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of matrix multiplication.
    
    Args:
        A (torch.Tensor): Input matrix of shape (M, K)
        B (torch.Tensor): Input matrix of shape (K, N)
    
    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    
    Explanation:
        1. Validate input dimensions
        2. Preallocate output tensor C
        3. Calculate launch grid dimensions
        4. Launch Triton kernel with autotuned parameters
    """
    if A.is_cuda and B.is_cuda:
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
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
    else:
        raise ValueError("Input tensors must be on CUDA device")

def matmul_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of matrix multiplication.
    
    Args:
        A (torch.Tensor): Input matrix of shape (M, K)
        B (torch.Tensor): Input matrix of shape (K, N)
    
    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    
    Explanation:
        1. Directly uses torch.matmul for computation
    """
    return torch.matmul(A, B)

# Test configurations
M_sizes = [256, 512, 1024, 2048, 4096]
N_sizes = [256, 512, 1024, 2048, 4096]
K_sizes = [256, 512, 1024, 2048, 4096]

def input_generator(M, N, K, device='cuda', dtype=torch.float16):
    """Generates normalized inputs for matmul verification"""
    A = torch.randn(M, K, device=device, dtype=dtype)
    A = torch.nn.functional.normalize(A, dim=-1)
    B = torch.randn(K, N, device=device, dtype=dtype)
    B = torch.nn.functional.normalize(B, dim=-1)
    return A, B

# Run verification and benchmarking
if __name__ == "__main__":
    verify_correctness(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
