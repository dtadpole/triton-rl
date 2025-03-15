# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

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
        a_ptr (tl.tensor): Pointer to matrix A. [Shape: (M, K)]
        b_ptr (tl.tensor): Pointer to matrix B. [Shape: (K, N)]
        c_ptr (tl.tensor): Pointer to matrix C. [Shape: (M, N)]
        M (int): Number of rows in matrix A and C.
        N (int): Number of columns in matrix B and C.
        K (int): Number of columns in matrix A and rows in matrix B.
        stride_am (int): Stride of matrix A in the M dimension.
        stride_ak (int): Stride of matrix A in the K dimension.
        stride_bk (int): Stride of matrix B in the K dimension.
        stride_bn (int): Stride of matrix B in the N dimension.
        stride_cm (int): Stride of matrix C in the M dimension.
        stride_cn (int): Stride of matrix C in the N dimension.
        BLOCK_SIZE_M (tl.constexpr): Size of block in the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Size of block in the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Size of block in the K dimension.
        GROUP_SIZE_M (tl.constexpr): Group size in the M dimension for cooperative fetching.

    Returns:
        None

    Explanation:
        1. Define the row and column indices for the output block.
        2. Offset the indices by the block ID to get the global indices.
        3. Initialize the accumulator to zero.
        4. Iterate over the K dimension in blocks.
        5. Load the A and B blocks into registers, masking out-of-bounds accesses.
        6. Perform the matrix multiplication using `tl.dot`.
        7. Write the result to the output block, masking out-of-bounds accesses.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Define the row and column indices for the output block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Initialize the accumulator to zero
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over the K dimension in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the A and B blocks into registers
        k_start = k * BLOCK_SIZE_K
        offs_ak = k_start + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_ak[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a_mask = (offs_am[:, None] < M) & (offs_ak[None, :] < K)
        b_mask = (offs_ak[:, None] < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

        # Perform the matrix multiplication
        acc += tl.dot(a, b)

    # Write the result to the output block
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def triton_matmul(a, b):
    """
    Description: Wraps the Triton kernel for matrix multiplication.

    Args:
        a (torch.Tensor): Matrix A. [Shape: (M, K)]
        b (torch.Tensor): Matrix B. [Shape: (K, N)]

    Returns:
        torch.Tensor: Matrix C (A @ B). [Shape: (M, N)]

    Explanation:
        1. Allocate memory for the output matrix C.
        2. Define the grid of blocks for the Triton kernel.
        3. Launch the Triton kernel.
        4. Return the output matrix C.
    """
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=2,
    )
    return c

def torch_matmul(a, b):
    """
    Description: Performs matrix multiplication using PyTorch.

    Args:
        a (torch.Tensor): Matrix A. [Shape: (M, K)]
        b (torch.Tensor): Matrix B. [Shape: (K, N)]

    Returns:
        torch.Tensor: Matrix C (A @ B). [Shape: (M, N)]

    Explanation:
        1. Use torch.matmul to perform matrix multiplication.
        2. Return the result.
    """
    return torch.matmul(a, b)

generated_triton_func = triton_matmul
generated_torch_func = torch_matmul

if __name__ == "__main__":
    import torch
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
