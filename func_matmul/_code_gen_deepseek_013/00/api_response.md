API: deepseek
Test Case: ./func_matmul/
Timestamp: 20250315_075436

<think>
Okay, so I need to generate Triton code for matrix multiplication with autotuning for block sizes and super blocks. Let me start by recalling how matrix multiplication works in Triton. The key is to tile the matrices into blocks and compute each block efficiently using the GPU's resources. Autotuning will help find the best block sizes for different matrix dimensions.

First, the user specified using autotune for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M. Triton's autotune decorator allows specifying configs for these parameters. I'll need to define a list of possible values for each block size and let Triton's autotuner select the best ones during compilation.

The Triton kernel function matmul_kernel will take pointers to matrices A and B, and output matrix C. Each program instance (block) will compute a tile of C. The input matrices are 2D, so their shapes are (M, K) for A and (K, N) for B. The output C is (M, N).

In the kernel, I'll need to compute the starting indices for each block. Using pid to determine the block's position, considering GROUP_SIZE_M for super blocks to improve L2 cache hit rate. Then, load the blocks of A and B into shared memory (using tl.dot), perform the matrix multiplication, and accumulate the result into C.

The Python wrapper function matmul will handle the kernel launch. It needs to compute the grid size based on M, N, and block sizes, then call the kernel with the autotuned config. Also, the torch function matmul_torch is needed for correctness checking against the Triton implementation.

In the main section, the provided code uses verify_correctness_func and benchmark_performance_func from benchmark.py. The input generator normalizes A and B to help with numerical stability during verification.

But wait, how do I structure the autotune configs? Looking at Triton's examples, the @triton.autotune decorator takes a configs list where each config specifies the block sizes and other parameters. I'll define possible values for each block size, perhaps using powers of two. For example, BLOCK_SIZE_M could be 64, 128, etc. GROUP_SIZE_M is typically 8 to balance occupancy and cache usage.

Also, the kernel should handle different data types, but the input generator uses float16. The kernel should be compatible with that. The tl.dot instruction in Triton handles different data types, so specifying fp16 in the kernel should work.

Another consideration is the computation of the number of blocks. Grid dimensions are M // BLOCK_SIZE_M and N // BLOCK_SIZE_N, but need to handle cases where they don't divide evenly by using triton.cdiv (ceiling division).

Wait, the kernel uses tl.dot, but for tiling, we need to load blocks of A and B into shared memory. Let me think: each block in Triton can load a tile of A (BLOCK_SIZE_M x BLOCK_SIZE_K) and a tile of B (BLOCK_SIZE_K x BLOCK_SIZE_N), then compute their product. Accumulating over K dimension.

So the steps in the kernel would be:

1. Determine the block's starting row and column in C.
2. Iterate over K dimension in chunks of BLOCK_SIZE_K.
3. Load the corresponding tiles from A and B.
4. Compute the dot product using tl.dot and accumulate.
5. Write the result to C.

But how to handle the autotuned parameters? The configs in the autotune decorator will set BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M. The kernel uses these variables as part of the block configuration.

Now, putting it all together in code. The matmul_triton.py file will have the kernel function with the autotune decorator, the wrapper function, the torch function, and the main block as specified.

Testing the code will involve running the script, which triggers the verification and benchmark functions. The test command would be something like 'python matmul_triton.py', assuming all dependencies are installed.

I should also make sure that the kernel uses the correct memory offsets and strides. For example, A is (M, K), so the stride for K is 1 if row-major. Triton uses column-major by default? Wait, in PyTorch, tensors are row-major, so Triton should access them correctly. Using pointers with the correct strides is important.

Overall, the main parts are: defining the autotune configs, writing the kernel with proper tiling and accumulation, and setting up the wrapper functions correctly. Also, ensuring that the input generator and test calls match the expected parameters.
</think>

```python
# filename: matmul_triton.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
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
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Performs matrix multiplication C = A @ B using Triton with autotuning.
    
    Args:
        a_ptr (tl.tensor): Pointer to input matrix A of shape (M, K)
        b_ptr (tl.tensor): Pointer to input matrix B of shape (K, N)
        c_ptr (tl.tensor): Pointer to output matrix C of shape (M, N)
        M (int): Number of rows in matrix A and C
        N (int): Number of columns in matrix B and C
        K (int): Number of columns in A and rows in B
        stride_am (int): Stride along M dimension for matrix A
        stride_ak (int): Stride along K dimension for matrix A
        stride_bk (int): Stride along K dimension for matrix B
        stride_bn (int): Stride along N dimension for matrix B
        stride_cm (int): Stride along M dimension for matrix C
        stride_cn (int): Stride along N dimension for matrix C
        BLOCK_SIZE_M (tl.constexpr): Autotuned block size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Autotuned block size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Autotuned block size for K dimension
        GROUP_SIZE_M (tl.constexpr): Number of blocks to schedule together
    
    Returns:
        None: Output is stored in c_ptr
    
    Explanation:
        1. Calculate block ID and split into super blocks using GROUP_SIZE_M
        2. Initialize offsets for A and B blocks
        3. Iterate over K dimension in BLOCK_SIZE_K chunks
        4. Load blocks of A and B into SRAM using mask for boundary checks
        5. Compute matrix multiplication with tl.dot and accumulate results
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

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton matrix multiplication wrapper function.
    
    Args:
        a (torch.Tensor): Input tensor of shape (M, K)
        b (torch.Tensor): Input tensor of shape (K, N)
    
    Returns:
        torch.Tensor: Output tensor of shape (M, N)
    
    Explanation:
        1. Verify input dimensions match matrix multiplication requirements
        2. Preallocate output tensor
        3. Calculate grid size based on block dimensions
        4. Launch optimized Triton kernel with autotuned parameters
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c

def matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    PyTorch matrix multiplication reference implementation.
    
    Args:
        a (torch.Tensor): Input tensor of shape (M, K)
        b (torch.Tensor): Input tensor of shape (K, N)
    
    Returns:
        torch.Tensor: Output tensor of shape (M, N)
    
    Explanation:
        1. Directly uses PyTorch's matrix multiplication operator
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
    verify_correctness_func(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
```

<test>
To verify correctness and benchmark performance, run:

```bash
python matmul_triton.py
```
This will:
1. Validate numerical correctness across multiple matrix sizes
2. Measure performance against PyTorch implementation
3. Generate JSON reports and performance comparison plots
</test>