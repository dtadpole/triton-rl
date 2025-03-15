# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Description: Performs matrix multiplication C = A @ B using Triton.

    Args:
        A (torch.Tensor): Pointer to the first input matrix. [Shape: (M, K)]
        B (torch.Tensor): Pointer to the second input matrix. [Shape: (K, N)]
        C (torch.Tensor): Pointer to the output matrix. [Shape: (M, N)]
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride in elements along the M dimension for A.
        stride_ak (int): Stride in elements along the K dimension for A.
        stride_bk (int): Stride in elements along the K dimension for B.
        stride_bn (int): Stride in elements along the N dimension for B.
        stride_cm (int): Stride in elements along the M dimension for C.
        stride_cn (int): Stride in elements along the N dimension for C.
        BLOCK_SIZE_M (int): Size of block in M dimension.
        BLOCK_SIZE_N (int): Size of block in N dimension.
        BLOCK_SIZE_K (int): Size of block in K dimension.
        GROUP_SIZE_M (int): Size of group in M dimension.

    Returns:
        None

    Explanation:
        1.  Compute the block ID along the M and N dimensions.
        2.  Compute the row and column indices for the current block.
        3.  Create a range of K indices for the inner loop.
        4.  Initialize the accumulator to zero.
        5.  Iterate over the K dimension, loading blocks of A and B, and performing the matrix multiplication.
        6.  Write the result to the output matrix C.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_id = pid // num_pid_n
    group_size_m = GROUP_SIZE_M
    pid_m = group_id // group_size_m
    pid_n = pid % num_pid_n

    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    b_ptrs = B + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_ptrs)
        b_block = tl.load(b_ptrs)

        accumulator += tl.dot(a_block, b_block)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn, accumulator, mask=mask)


def generated_triton_func(a, b):
    """
    Description: Wraps the Triton kernel to perform matrix multiplication.

    Args:
        a (torch.Tensor): The first input matrix. [Shape: (M, K)]
        b (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The output matrix. [Shape: (M, N)]

    Explanation:
        1.  Allocate memory for the output matrix.
        2.  Launch the Triton kernel with appropriate grid size and arguments.
        3.  Return the output matrix.
    """
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    _kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c


def generated_torch_func(a, b):
    """
    Description: Performs matrix multiplication using torch.

    Args:
        a (torch.Tensor): The first input matrix. [Shape: (M, K)]
        b (torch.Tensor): The second input matrix. [Shape: (K, N)]

    Returns:
        torch.Tensor: The output matrix. [Shape: (M, N)]

    Explanation:
        1.  Perform matrix multiplication using torch.matmul.
        2.  Return the output matrix.
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
