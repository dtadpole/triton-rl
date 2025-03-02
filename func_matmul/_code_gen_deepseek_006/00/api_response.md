API: deepseek
Test Case: ./func_matmul/
Timestamp: 20250302_090059

<think>
Okay, I need to generate Triton code for matrix multiplication with autotuning. Let's start by recalling how Triton's matrix multiplication works. The user specified using autotune for block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) and super blocks with GROUP_SIZE_M. 

First, the Triton kernel function. The kernel should handle tiling the input matrices into blocks. Autotuning requires defining configs with different block sizes. I remember that Triton's autotune uses the @triton.autotune decorator with a configs list. Each config specifies different block sizes and GROUP_SIZE_M. 

The kernel function will have parameters for the pointers to matrices A, B, and C, along with their dimensions M, N, K, and strides. Inside the kernel, I need to compute the starting indices for each block and then perform the matrix multiplication using those blocks. 

For the autotuning part, I should define several possible block sizes. Common choices are powers of two, like 64, 128, 256. The GROUP_SIZE_M is used for super-blocking to improve memory access patterns. The configs will combine different BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M values.

Next, the Python wrapper function matmul_triton. It takes A and B, checks their shapes, initializes the output matrix C, then calls the kernel with the appropriate grid and configurations. The kernel is launched with the grid size determined by dividing M, N by the block sizes, but since those are autotuned, maybe the grid is calculated using triton.cdiv and the block sizes from the config. Wait, but in the kernel launch, the grid is (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)), which depends on the current config's block sizes. That makes sense because each config has different block sizes, so the grid varies accordingly.

Then, the user wants to test correctness and performance using verify_correctness and benchmark_performance functions from benchmark.py. The input generator creates random matrices A and B normalized along the last dimension. The generated_triton_func is the matmul_triton function, and the generated_torch_func is torch.matmul. The M_sizes, N_sizes, K_sizes are given as lists of sizes.

I need to make sure the Triton kernel uses proper pointer arithmetic and matrix strides. The explanation in the docstring should outline each step the kernel takes: computing block indices, initializing the accumulator, iterating over K dimension with block tiles, loading data into shared memory, computing the dot product, and storing the result.

Wait, in Triton, shared memory isn't explicitly used like in CUDA. Instead, the compiler manages memory accesses. So the steps would involve loading blocks from A and B into registers or local memory, then performing the dot product. Maybe the explanation should mention using tl.dot for the matrix multiplication step.

Also, for the autotune configs, how many different combinations should I include? The example uses four different block sizes for each dimension and a couple of group sizes. For instance, BLOCK_SIZE_M and N could be 64, 128, etc., and GROUP_SIZE_M 8. But to cover more possibilities, maybe multiple combinations of M, N, K block sizes.

Another point: when using autotune, the kernel is compiled multiple times with different configurations, and the best-performing one is selected based on the target hardware. So the configs should include a variety of tile sizes that are likely to perform well on different GPU architectures.

Testing the code would involve running the provided benchmark commands. The user provided the code to include, which uses the input generator with various sizes. The verify_correctness function checks if the Triton implementation matches the PyTorch result, and benchmark_performance compares their speeds.

Putting this all together, the Triton kernel function will be decorated with autotune, have configs defined, and the wrapper function sets up the tensors and launches the kernel. The comments for each function need to follow the specified structure, detailing arguments, returns, and explanation steps.

I should also make sure that the Triton kernel uses the correct memory ordering and strides. For example, when accessing matrix A, the row is determined by the program ID along M, and similarly for B along N. The block indices need to be computed correctly, considering the GROUP_SIZE_M for super-blocking to optimize data reuse.

In the explanation section of the kernel's docstring, each step should clearly outline the process: determining block indices, initializing accumulator, loop over K, loading A and B tiles, performing the dot product, and storing the result.

Finally, the test section includes the bash command to run the script, assuming the necessary dependencies are installed. The user mentioned not to include pip install commands, so just the python command to execute the generated file.
</think>

```python
# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=8, num_warps=4),
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
    Performs matrix multiplication using Triton with autotuned block configurations.

    Args:
        a_ptr (triton.Pointer): Pointer to input matrix A. Shape: (M, K)
        b_ptr (triton.Pointer): Pointer to input matrix B. Shape: (K, N)
        c_ptr (triton.Pointer): Pointer to output matrix C. Shape: (M, N)
        M (int): Number of rows in matrix A and output matrix C
        N (int): Number of columns in matrix B and output matrix C
        K (int): Number of columns in matrix A and rows in matrix B
        stride_am (int): Stride between rows in matrix A
        stride_ak (int): Stride between columns in matrix A
        stride_bk (int): Stride between rows in matrix B
        stride_bn (int): Stride between columns in matrix B
        stride_cm (int): Stride between rows in matrix C
        stride_cn (int): Stride between columns in matrix C
        BLOCK_SIZE_M (tl.constexpr): Autotuned block size for M dimension
        BLOCK_SIZE_N (tl.constexpr): Autotuned block size for N dimension
        BLOCK_SIZE_K (tl.constexpr): Autotuned block size for K dimension
        GROUP_SIZE_M (tl.constexpr): Group size for super-blocking optimization

    Returns:
        None: Output is stored through c_ptr

    Explanation:
        1. Calculate block indices for M and N dimensions using program ID
        2. Super-blocking: Group nearby blocks in M dimension to improve data reuse
        3. Calculate offsets for A and B matrices based on block indices
        4. Initialize accumulator with zeros
        5. Loop over K dimension in blocks, accumulating matrix products
        6. Load blocks from A and B matrices into local memory
        7. Compute dot product using Triton's matrix operations
        8. Store final result to output matrix C
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
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator.to(tl.float16))

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton matrix multiplication wrapper function.

    Args:
        A (torch.Tensor): Input matrix A. Shape: (M, K)
        B (torch.Tensor): Input matrix B. Shape: (K, N)

    Returns:
        torch.Tensor: Output matrix C. Shape: (M, N)

    Explanation:
        1. Verify input dimensions match matrix multiplication requirements
        2. Preallocate output matrix C with zeros
        3. Calculate grid size based on input dimensions and block sizes
        4. Launch Triton kernel with autotuned configurations
        5. Return the computed result matrix
    """
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C

def torch_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for PyTorch matmul with same interface as triton version"""
    return torch.matmul(A, B)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    M_sizes = [256, 512, 1024, 2048, 4096]
    N_sizes = [256, 512, 1024, 2048, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B
    
    verify_correctness(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, torch_matmul, input_generator, [M_sizes, N_sizes, K_sizes])
```

<test>
To verify correctness and benchmark performance, run:

```bash
python matmul_triton.py
```
</test>