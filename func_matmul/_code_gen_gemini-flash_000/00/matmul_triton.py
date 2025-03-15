# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 128 and args['N'] > 256),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 128),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 32 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 16 and args['N'] > 32),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 128 and args['N'] > 128),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 8, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 32 and args['N'] > 32),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 4, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 16 and args['N'] > 16),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 128 and args['N'] > 64),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, prehook=lambda args: args['M'] > 64 and args['N'] > 32),
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
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Description: Performs matrix multiplication C = A @ B using Triton.

    Args:
        a_ptr (torch.Tensor): Pointer to the first input matrix A. Shape: (M, K)
        b_ptr (torch.Tensor): Pointer to the second input matrix B. Shape: (K, N)
        c_ptr (torch.Tensor): Pointer to the output matrix C. Shape: (M, N)
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride in elements along the M dimension for A.
        stride_ak (int): Stride in elements along the K dimension for A.
        stride_bk (int): Stride in elements along the K dimension for B.
        stride_bn (int): Stride in elements along the N dimension for B.
        stride_cm (int): Stride in elements along the M dimension for C.
        stride_cn (int): Stride in elements along the N dimension for C.
        BLOCK_SIZE_M (tl.constexpr): Size of the block in the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Size of the block in the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Size of the block in the K dimension.
        GROUP_SIZE_M (tl.constexpr): Group size for the M dimension.

    Returns:
        None

    Explanation:
        1.  Compute the starting offsets for the blocks in A and B.
        2.  Create a range of offsets for the M, N, and K dimensions within the block.
        3.  Initialize an accumulator to zero.
        4.  Iterate over the K dimension to perform the matrix multiplication.
        5.  Load the blocks from A and B into registers.
        6.  Perform the dot product of the blocks.
        7.  Write the result to the output matrix C.
    """
    # 1. Compute block ID
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 2. Compute data offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. Create mask
    mask_am = offs_am[:, None] < M
    mask_bn = offs_bn[None, :] < N

    # 4. Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 5. Iterate and compute
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B
        a = tl.load(a_ptrs, mask=mask_am & (offs_k[None, :] + k < K), other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=mask_bn & (offs_k[:, None] + k < K), other=0.0).to(tl.float16)

        # Dot product
        accumulator += tl.dot(a, b)

        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 6. Write back to C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_cm = offs_cm[:, None] < M
    mask_cn = offs_cn[None, :] < N
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask_cm & mask_cn)

def matmul(a, b):
    """
    Description: Wraps the triton kernel to perform matrix multiplication.

    Args:
        a (torch.Tensor): The first input matrix. Shape: (M, K)
        b (torch.Tensor): The second input matrix. Shape: (K, N)

    Returns:
        torch.Tensor: The output matrix. Shape: (M, N)

    Explanation:
        1.  Allocate memory for the output matrix.
        2.  Determine the grid size for the Triton kernel.
        3.  Launch the Triton kernel.
        4.  Return the output matrix.
    """
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c

def torch_matmul(a, b):
    """
    Description: Performs matrix multiplication using torch.

    Args:
        a (torch.Tensor): The first input matrix.
        b (torch.Tensor): The second input matrix.

    Returns:
        torch.Tensor: The output matrix.
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
    verify_correctness_func(matmul, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(matmul, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
