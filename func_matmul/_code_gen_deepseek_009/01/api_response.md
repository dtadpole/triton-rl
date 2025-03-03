API: deepseek
Test Case: ./func_matmul/
Timestamp: 20250302_235921

<think>
Okay, I need to generate a Triton kernel for matrix multiplication with autotuning for block sizes and super blocks. Let me start by recalling how Triton's autotuning works. Autotune requires defining a config with parameters that can vary, like BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M. The configs will be tried during autotuning to find the best-performing configuration.

The kernel function will use these block sizes. The user mentioned using super blocks, which probably refers to grouping multiple blocks together for better memory access patterns. Oh, right, GROUP_SIZE_M is used for that. So in the Triton kernel, when splitting the M dimension, we might loop over groups of blocks.

The Triton matmul kernel typically works by dividing the matrices into blocks. Each block computes a portion of the output matrix. For each element in the block, we accumulate the dot product over K dimension by iterating through blocks of K.

I need to set up the grid for the kernel. The grid dimensions would be based on the number of blocks in M and N directions. For example, grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)).

Inside the kernel, we'll have to manage pointers to the matrices A and B. A is (M, K) and B is (K, N). The output C is (M, N). Each program loads a block from A and a block from B, multiplies them, and accumulates into the corresponding block of C.

For autotuning, I'll use the @triton.autotune decorator, specifying the configs. Each config will have different combinations of BLOCK sizes and GROUP_SIZE_M. The possible values for block sizes are typically powers of two, like 16, 32, 64, etc. The GROUP_SIZE_M could be 8, which is a common choice to reduce synchronization between warps.

The Python function matmul_triton will take A and B as inputs, check their dimensions, allocate the output tensor C, and invoke the kernel with the appropriate grid and config.

Then, the generated_torch_func is just torch.matmul. But since the user provided a generated_torch_func parameter, maybe I should wrap it in a lambda that handles the inputs, like lambda A, B: A @ B.

Wait, in the test code, verify_correctness and benchmark_performance will compare the Triton function against the torch function. So the Triton function needs to be a callable that takes A and B and returns C. So the generated_triton_func is the matmul_triton function.

I need to make sure that the Triton kernel handles different data types, but the input generator uses float16. So the kernel should work with float16 inputs and accumulate in float32 to prevent precision loss. Wait, in the kernel, the accumulator is initialized as zeros with dtype=torch.float32. That's correct.

Now, the explanation section for the kernel function. The steps would be: 1. Get the program ID for rows and columns. 2. Compute the block indices for M, N. 3. Initialize pointers for A and B blocks. 4. Loop over K in blocks, accumulate products. 5. Write the result to C.

For the Python function matmul_triton, the explanation steps are: 1. Check input dimensions. 2. Allocate output. 3. Define kernel configuration. 4. Launch kernel. 5. Return result.

I should also include the test code as specified. But the user says to generate the main code and call the functions from benchmark.py, which already exists. So in the code, after defining the functions, call verify_correctness and benchmark_performance with the generated functions and input generator.

Wait, but the user's test code includes M_sizes, N_sizes, K_sizes, and the input generator. So in the generated matmul_triton.py, after defining the functions, those test calls should be included. But the benchmark.py has the functions, so I can import them.

Putting it all together, the matmul_triton.py will have the Triton kernel function, the Python wrapper, the generated_triton_func as the wrapper, and the generated_torch_func as torch.matmul. Then, the input_generator function is defined, and then the calls to verify and benchmark.

But wait, the user's code example shows that they pass generated_triton_func, generated_torch_func, input_generator, and the sizes. So in the code, those functions need to be properly named. For example:

generated_triton_func = matmul_triton
generated_torch_func = torch.matmul

But since the input generator uses normalize, the test cases will cover those sizes.

Now, possible issues: The Triton kernel's grid might not handle cases where M or N is not divisible by the block size. But using triton.cdiv in the grid setup handles that. Also, the kernel uses offsets correctly with masks to avoid out-of-bounds access.

Testing this would require running the script, which would call verify_correctness with different matrix sizes, checking that the Triton implementation matches Torch's results within a tolerance. Then benchmark the performance.

The autotune configs are set up with possible block sizes and group sizes. The kernel uses those parameters to determine the tiling.

Another point: the kernel uses tl.dot, which efficiently computes the block matrix product. The accumulation in acc ensures that the partial sums are correctly added.

So putting all this into the code with proper comments as per the structure.
</think>

```python
# filename: matmul_triton.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
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
        A_ptr (triton.Pointer): Pointer to 2D input tensor A of shape (M, K)
        B_ptr (triton.Pointer): Pointer to 2D input tensor B of shape (K, N)
        C_ptr (triton.Pointer): Pointer to 2D output tensor C of shape (M, N)
        M (int): Number of rows in matrix A and C
        N (int): Number of columns in matrix B and C
        K (int): Number of columns in A and rows in B
        stride_am (int): Stride along M dimension for A
        stride_ak (int): Stride along K dimension for A
        stride_bk (int): Stride along K dimension for B
        stride_bn (int): Stride along N dimension for B
        stride_cm (int): Stride along M dimension for C
        stride_cn (int): Stride along N dimension for C
        BLOCK_SIZE_M (tl.constexpr): Tuning parameter for block size in M dimension
        BLOCK_SIZE_N (tl.constexpr): Tuning parameter for block size in N dimension
        BLOCK_SIZE_K (tl.constexpr): Tuning parameter for block size in K dimension
        GROUP_SIZE_M (tl.constexpr): Group size for super-blocking in M dimension

    Returns:
        None: Output is written through C_ptr

    Explanation:
        1. Calculate program ID for current computation block
        2. Calculate block ranges for M and N dimensions using super-blocking
        3. Initialize pointers for A and B matrices
        4. Iterate over K dimension in blocks, accumulating partial sums
        5. Store final accumulated results to output matrix C
    """
    pid = tl.program_id(0)
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
    a_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptr, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk
    
    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr, c, mask=mask)

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    TRITON-accelerated matrix multiplication wrapper function.

    Args:
        A (torch.Tensor): Input tensor of shape (M, K)
        B (torch.Tensor): Input tensor of shape (K, N)

    Returns:
        torch.Tensor: Output tensor of shape (M, N)

    Explanation:
        1. Validate input dimensions match matrix multiplication requirements
        2. Preallocate output tensor C with appropriate shape
        3. Calculate grid size based on input dimensions and block configuration
        4. Launch optimized kernel with autotuned parameters
    """
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C

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

    verify_correctness(matmul_triton, torch.matmul, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance(matmul_triton, torch.matmul, input_generator, [M_sizes, N_sizes, K_sizes])
```

<test>
```bash
python matmul_triton.py
```
</test>