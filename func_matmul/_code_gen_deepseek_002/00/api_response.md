API: deepseek
Test Case: ./func_matmul/
Timestamp: 20250302_052532

<think>
Okay, I need to generate a Triton kernel for matrix multiplication with auto-tuning for block sizes and GROUP_SIZE_M. Let's start by recalling how Triton's matrix multiplication works. The basic idea is to divide the matrices into blocks and compute each block using multiple threads. Autotuning requires specifying configurable parameters and letting Triton test different combinations to find the best performance.

First, the Triton kernel should handle the block matrix multiplication. The block sizes M, N, K (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) need to be autotuned. Also, GROUP_SIZE_M is used for grouping blocks along the M dimension to improve L2 cache hit rate. These parameters should be set with triton.Config, and then autotuned.

I remember that in Triton, the @triton.autotune decorator is used with a list of configs. Each config specifies different values for the block sizes and group size. The kernel function will have these as parameters with default values, but the autotuner will override them during tuning.

Next, the kernel function's structure: Each program instance handles a block. The pointers for matrices A, B, and C are initialized. We need to compute the row and column indices for the block. Then, iterate over K dimension in chunks of BLOCK_SIZE_K, loading the corresponding blocks of A and B into shared memory (using tl.dot), and accumulate the result into a temporary variable. Finally, write the accumulated result to the output matrix C.

The torch function is straightforward: use torch.matmul. But I need to make sure the inputs are in the correct format. The input generator normalizes A along the last dimension and B along the first dimension. So the generated_torch_func is just a lambda that takes A, B and returns A @ B.

For the main code, I need to import necessary modules: torch, triton, and from benchmark import verify_correctness and benchmark_performance. Then define the Triton kernel with the autotuning configs, the torch function, and the input generator as specified. Finally, call verify_correctness and benchmark_performance with the input generator and the size lists.

Wait, the input generator's arguments are M, N, K, so the sizes in verify_correctness are given as [M_sizes, N_sizes, K_sizes]. Each combination of M, N, K from these lists will be tested.

Now, let's structure the Triton kernel. The kernel will be annotated with @triton.jit and include the autotune decorator. The function signature should take A, B, C, M, N, K as parameters, along with the block sizes and group size. Inside the kernel, compute the PID, determine the blocks, and perform the matrix multiplication with loop over K.

I need to make sure that the block sizes are correctly used in the kernel. For example, BLOCK_SIZE_M is the block size along the M dimension of A and C. The GROUP_SIZE_M groups multiple blocks along M to reuse B's data from the L2 cache.

Testing the code involves running the Python script, which will execute the verification and benchmarking. The bash command would be something like 'python main.py', assuming the code is saved in main.py.

Potential issues: Making sure the Triton kernel correctly handles the matrix dimensions and that the autotune configurations are properly set. Also, ensuring that the tl.dot operation correctly multiplies the blocks. Another thing is handling different data types, but the input generator uses float16, so the Triton kernel should use the same precision.

Wait, in the Triton kernel, the data types for pointers are important. A and B are of type float16, so the accumulators should be float32. The kernel uses tl.dot(a, b) which accumulates in float32, then converts to float16 when storing in C. But since the problem says to test correctness against the torch function, which is fp16, the Triton kernel's output should be fp16. So the accumulator is fp32, then cast to fp16 when stored.

I think that's the standard approach to maintain precision. The Triton kernel's computation is done in higher precision and then stored as lower precision.

Now, putting it all together. The Triton kernel function is named 'matmul_kernel', and the main function wraps it into a callable that takes A and B, creates C, and launches the kernel.

The generated_triton_func is a Python function that takes A, B and returns the output tensor C. The generated_torch_func is lambda A, B: A @ B.

Finally, the main code calls verify_correctness and benchmark_performance with these functions and the input generator.

Testing this code would require installing Triton, PyTorch, and CUDA dependencies. But the user mentioned not to include pip install commands, so the test command is simply running the script.
</think>

```python
# filename: matmul_triton.py
import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Performs matrix multiplication using Triton with auto-tuned block configurations.

    Args:
        A_ptr (triton.PointerType): Pointer to input matrix A. Shape (M, K)
        B_ptr (triton.PointerType): Pointer to input matrix B. Shape (K, N)
        C_ptr (triton.PointerType): Pointer to output matrix C. Shape (M, N)
        M (tl.constexpr): Number of rows in matrix A
        N (tl.constexpr): Number of columns in matrix B
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B
        stride_am (tl.constexpr): Stride between rows of matrix A
        stride_ak (tl.constexpr): Stride between columns of matrix A
        stride_bk (tl.constexpr): Stride between rows of matrix B
        stride_bn (tl.constexpr): Stride between columns of matrix B
        stride_cm (tl.constexpr): Stride between rows of matrix C
        stride_cn (tl.constexpr): Stride between columns of matrix C
        BLOCK_SIZE_M (tl.constexpr): Tuned block size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Tuned block size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Tuned block size for K dimension
        GROUP_SIZE_M (tl.constexpr): Group size for L2 cache optimization

    Returns:
        None: Output is written directly to memory via C_ptr

    Explanation:
        1. Calculate program ID and determine block ranges using super-block grouping
        2. Initialize pointers for input matrices with double buffering
        3. Iterate through K dimension in chunks of BLOCK_SIZE_K
        4. Load blocks of A and B matrices into SRAM
        5. Compute matrix product using tl.dot and accumulate results
        6. Write final accumulated results to output matrix C
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
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for Triton matrix multiplication kernel.

    Args:
        A (torch.Tensor): Input matrix A of shape (M, K)
        B (torch.Tensor): Input matrix B of shape (K, N)

    Returns:
        torch.Tensor: Output matrix C of shape (M, N)

    Explanation:
        1. Validate input dimensions
        2. Pre-allocate output tensor
        3. Calculate grid size based on input dimensions and block sizes
        4. Launch optimized Triton kernel
    """
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda opt: (triton.cdiv(M, opt.BLOCK_SIZE_M) * triton.cdiv(N, opt.BLOCK_SIZE_N),)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        GROUP_SIZE_M=8
    )
    return C

# Define test configurations
M_sizes = [512, 1024, 2048, 4096]
N_sizes = [512, 1024, 2048, 4096]
K_sizes = [512, 1024, 2048, 4096]

def input_generator(M, N, K, device='cuda', dtype=torch.float16):
    """Generates normalized input matrices for testing"""
    A = torch.randn(M, K, device=device, dtype=dtype)
    A = torch.nn.functional.normalize(A, dim=-1)
    B = torch.randn(K, N, device=device, dtype=dtype)
    B = torch.nn.functional.normalize(B, dim=0)
    return A, B

if __name__ == "__main__":
    # Test correctness and benchmark performance
    verify_correctness(triton_matmul, 
                      lambda A, B: A @ B, 
                      input_generator,
                      [M_sizes, N_sizes, K_sizes])
    
    benchmark_performance(triton_matmul,
                         lambda A, B: A @ B,
                         input_generator,
                         [M_sizes, N_sizes, K_sizes])
```

<test>
```bash
python matmul_triton.py
```
</test>