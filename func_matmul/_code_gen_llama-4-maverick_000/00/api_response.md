Timestamp: 20250406_051446

<think>
The task is to generate Triton code for Matrix Multiplication and test its correctness and performance. The code should be in a file named `matmul_triton.py`. The given instructions specify that we should use autotune for block sizes and test the code using the `verify_correctness_func` and `benchmark_performance_func` functions from the `benchmark.py` file.

First, let's understand the requirements:
1. We need to implement a Triton kernel for matrix multiplication.
2. We should use autotuning for block sizes (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`) and `GROUP_SIZE_M`.
3. The code should be tested for correctness and performance against a PyTorch implementation.

To achieve this, we'll follow these steps:
1. Implement the Triton kernel for matrix multiplication with autotuning.
2. Define a Python function that wraps the Triton kernel.
3. Test the correctness and performance using the provided functions from `benchmark.py`.

Let's start by designing the Triton kernel.

### Triton Kernel Design
The Triton kernel for matrix multiplication will:
- Take two input matrices `A` and `B` and compute their product `C = A @ B`.
- Use autotuning for `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, and `GROUP_SIZE_M`.
- Handle matrices of varying sizes.

### Implementation

We'll implement the Triton kernel and the wrapper function in `matmul_triton.py`.

```python
# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

# Define autotuning configurations for block sizes
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
    Description: Triton kernel for matrix multiplication C = A @ B.

    Args:
        A_ptr (triton.Pointer): Pointer to matrix A. [M, K]
        B_ptr (triton.Pointer): Pointer to matrix B. [K, N]
        C_ptr (triton.Pointer): Pointer to output matrix C. [M, N]
        M (int): Number of rows in A and C.
        N (int): Number of columns in B and C.
        K (int): Number of columns in A and rows in B.
        stride_am (int): Stride of A along rows.
        stride_ak (int): Stride of A along columns.
        stride_bk (int): Stride of B along rows.
        stride_bn (int): Stride of B along columns.
        stride_cm (int): Stride of C along rows.
        stride_cn (int): Stride of C along columns.

    Returns:
        None

    Explanation:
        1. The kernel is designed to compute C = A @ B using blocked matrix multiplication.
        2. It uses Triton's autotune feature to optimize block sizes and other parameters.
        3. The computation is divided into blocks of size BLOCK_SIZE_M x BLOCK_SIZE_N, with K dimension blocked by BLOCK_SIZE_K.
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
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(A_ptrs)
        B = tl.load(B_ptrs)
        accumulator += tl.dot(A, B)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk

    C = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(C_ptrs, C)

def matmul_triton(A, B):
    """
    Description: Python wrapper for Triton matrix multiplication kernel.

    Args:
        A (torch.Tensor): Input matrix A. [M, K]
        B (torch.Tensor): Input matrix B. [K, N]

    Returns:
        torch.Tensor: Output matrix C = A @ B. [M, N]

    Explanation:
        1. Checks input validity and extracts dimensions.
        2. Allocates output tensor C.
        3. Launches the Triton kernel to compute C = A @ B.
    """
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
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
        torch.Tensor: Output matrix C = A @ B. [M, N]
    """
    return A @ B

if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]

    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=0)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(matmul_triton, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(matmul_triton, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
```

</think>

```python
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
        triton': Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
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
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(A_ptrs)
        B = tl.load(B_ptrs)
        accumulator += tl.dot(A, B)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk

    C = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(C_ptrs, C)

def matmul_triton(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
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
    return A @ B

if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]

    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=0)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(matmul_triton, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(matmul_triton, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
```

<test>
To test the `matmul_triton.py` code, you can run it directly as it contains a `if __name__ == "__main__":` block that executes the testing and benchmarking.

```bash
python matmul_triton.py
```

This command will run the correctness test and performance benchmark for the Triton matrix multiplication implementation against the PyTorch implementation.
</test>