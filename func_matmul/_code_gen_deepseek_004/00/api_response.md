API: deepseek
Test Case: ./func_matmul/
Timestamp: 20250302_075114

<think>
Okay, I need to generate a Triton kernel for matrix multiplication with autotuning. Let me start by recalling how matrix multiplication works in Triton. The key is to tile the matrices into blocks and compute each block efficiently using the GPU's threads.

First, the user mentioned using autotune for block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) and super blocks with GROUP_SIZE_M. Oh right, Triton's autotune feature allows specifying configs with different parameters. So I should define a list of configurations that Triton can try during autotuning.

The triton.autotune decorator will take these configs. Each config specifies different block sizes and GROUP_SIZE_M. The autotuner will benchmark these to find the fastest configuration for each problem size.

Next, the Triton kernel itself. The input matrices A and B need to be accessed with pointers. The kernel will compute a block of the output matrix C. Each program instance handles a block of size BLOCK_SIZE_M x BLOCK_SIZE_N.

Steps in the kernel:
1. Calculate the row and column indices for the block in C.
2. Create pointers for the blocks in A and B. Since we need to loop over K dimension, we'll use multiple stages (BLOCK_SIZE_K steps).
3. Initialize the accumulator for the block in C.
4. Loop over K in steps of BLOCK_SIZE_K, loading blocks of A and B, then performing matrix multiplication on these blocks.
5. Use tl.dot to compute the product and accumulate the result.
6. After the loop, write the accumulated block back to C.

For correctness testing and benchmarking, the user provided code that uses verify_correctness and benchmark_performance functions from benchmark.py. So I need to write a PyTorch function (matmul_torch) that does the same operation for verification. The input generator normalizes A and B along their last dimension, which might help with numerical stability but I need to make sure that the Triton kernel and the PyTorch function handle the inputs the same way.

Wait, in the input_generator, A is (M, K) and normalized along dim=-1, which is K. Similarly for B (K, N) normalized along dim=-1 (N). Hmm, maybe that's a typo. Because in matrix multiplication, the inner dimension is K. But normalization along different axes might affect the product. However, the user's code is given, so perhaps I should proceed as per the provided input_generator.

But I need to make sure that the generated_triton_func and generated_torch_func correctly compute C = A @ B. Let me check that. The PyTorch function matmul_torch should simply perform torch.matmul(A, B). The Triton function matmul_triton does the same.

In the kernel, I'll need to handle different data types. The user's input uses torch.float16, so the Triton kernel should use tl.float16. Wait, but A and B are float16, but the accumulator should be float32 to avoid precision loss. So the kernel will cast the elements to float32 for computation.

In the code, the pointers for A and B are defined with tl.make_block_ptr. The strides are important here. For matrix A (M, K), the strides would be (K, 1) if it's row-major. Similarly for B (K, N), strides (N, 1) if column-major? Wait, wait, B is (K, N), so if we access B in column-major order, each element in a row would be consecutive. Wait, maybe I need to double-check.

Wait, in Triton, when using block pointers, the shape and strides matter. For matrix A (M, K), which is row-major, each row is contiguous. So the strides would be (K, 1). For matrix B (K, N), which is row-major (since it's generated normally), each row is contiguous. So when multiplying A (M,K) * B (K,N), the standard matrix multiplication is possible if B is (K, N). But to access B's columns efficiently, maybe B needs to be transposed. Wait, no. The Triton kernel for matmul usually accesses B as a column-major matrix. Wait, perhaps I need to transpose B first. Wait, no. Let me think again.

In standard matrix multiplication, the element C[i,j] is the dot product of A's i-th row and B's j-th column. So the kernel should iterate over K, and for each step, take a block from A's row and B's column.

Wait, no. A is (M, K), B is (K, N), the product is (M, N). To compute each element in the product, you take the dot product of a row from A and a column from B. However, in terms of memory layout, if B is stored in row-major, then a column in B is not contiguous. This can lead to inefficient memory access. So perhaps the kernel is designed to handle this by loading blocks in a way that allows efficient access.

Alternatively, the Triton kernel may use a tiling approach where each thread block processes a tile of the output matrix. For each tile, the kernel loads a block from A (BLOCK_SIZE_M x BLOCK_SIZE_K) and a block from B (BLOCK_SIZE_K x BLOCK_SIZE_N), then computes their product. This requires iterating over the K dimension in chunks of BLOCK_SIZE_K.

So the kernel's loop over K in steps of BLOCK_SIZE_K. For each step, load a block from A and a block from B, compute their dot product, and accumulate the result.

Now, regarding the use of tl.dot. The tl.dot function allows for efficient matrix multiplication of blocks. The input to tl.dot would be the loaded blocks from A and B, properly reshaped.

In the kernel, after initializing the accumulator, we loop over K in steps of BLOCK_SIZE_K. For each iteration, we compute the pointers to the current blocks in A and B. Then we load those blocks using tl.load. Wait, maybe using tl.advance to move the pointers?

Alternatively, using block pointers with offsets. The block pointers can be advanced along the K dimension. For example, the block in A starts at (start_m, start_k) and moves along K, while the block in B starts at (start_k, start_n) and moves along K.

Wait, maybe the block pointers for A are for rows in M and slices in K, and for B, columns in N and slices in K. For each iteration, the pointers to A and B are offset along K by BLOCK_SIZE_K. So A's block is (BLOCK_SIZE_M, BLOCK_SIZE_K), and B's block is (BLOCK_SIZE_K, BLOCK_SIZE_N). The tl.dot product of these two is a (BLOCK_SIZE_M, BLOCK_SIZE_N) block, which is accumulated into the result.

I think that's the approach. The code for the kernel would look something like:

Initialize the offsets for the current block in C (row and column).
Then, create pointers for A and B that move along K.
Loop over K, each step loading a block from A and B, compute the product, accumulate.

Now, the autotuning part: the configs should include different values for BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M. For example, common block sizes are 64, 128, 256. GROUP_SIZE_M is used for super-grouping to improve memory access patterns. For example, GROUP_SIZE_M=8 might group multiple M blocks together.

So the configs list would have entries like triton.Config({'BLOCK_SIZE_M': 64, ...}, ...). The num_warps parameter may vary with block sizes. For example, a block size of 128 might use 4 warps (128 threads, each warp is 32 threads, so 4 warps).

The @triton.autotune decorator will take these configs and select the best one for each input size.

Now, the generated_triton_func is a function that takes A and B and returns the product using the Triton kernel. The generated_torch_func is torch.matmul.

Wait, but the input_generator returns A and B as normalized. So the Triton kernel and the PyTorch function should compute the same product. The verify_correctness function would check whether the Triton kernel's output matches the PyTorch output.

In the code, the Triton function (matmul_triton) should take A and B, check their dimensions, possibly handle non-contiguous tensors by calling .contiguous(), and then launch the kernel.

Testing with M_sizes, N_sizes, K_sizes as [256, 512, 1024, 2048, 4096], the code will test various matrix sizes.

Putting this all together:

The matmul_triton.py file will contain:

- The Triton kernel with autotune configs.
- The wrapper function to call the kernel.
- The PyTorch reference function.
- The code to call verify_correctness and benchmark_performance.

Wait, but the user's test code says to call verify_correctness and benchmark_performance with generated_triton_func and generated_torch_func. So in the code, the main part should import the functions (matmul_triton and matmul_torch) and then run the tests.

But the user provided code shows that after defining the input_generator, they call verify_correctness with the functions and the M_sizes, etc. So the code in matmul_triton.py should include:

import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

Then, define the kernel and functions, then call the verify and benchmark functions.

Wait, but the user's code in the problem statement says that benchmark.py already exists and has the functions. So in the generated matmul_triton.py, after defining the kernel and the wrapper functions, the code should call verify_correctness and benchmark_performance with the appropriate parameters.

The input_generator is defined as per the user's code. Then, the M_sizes etc. are lists. So the code would:

M_sizes = [256, ...]
N_sizes = ...
K_sizes = ...
def input_generator(...): ...

Then:

verify_correctness(matmul_triton, matmul_torch, input_generator, [M_sizes, N_sizes, K_sizes])
benchmark_performance(... same ...)

Wait, but the user's code uses `generated_triton_func` and `generated_torch_func`, which in this case are the matmul_triton and matmul_torch functions.

So the steps are:

1. Define the Triton kernel with autotune.
2. Define the matmul_triton function that calls the kernel.
3. Define the matmul_torch function as a reference.
4. Define the input_generator as per user's code.
5. Call verify and benchmark functions with the matrices.

But wait, in the input_generator, the user's code takes M, N, K as parameters. The verify_correctness function is supposed to generate inputs for all combinations of M in M_sizes, N in N_sizes, K in K_sizes. So the code will test all those sizes.

Now, the kernel needs to handle different sizes, which it should as the kernel is designed with variable sizes, using the grid parameter in the launch.

Potential issues:

- Data type: the input_generator uses float16. The kernel should handle float16, using fp32 for accumulation. The Triton code should cast the loaded elements to float32 before computation.

- Handling of non-divisible sizes: if K is not divisible by BLOCK_SIZE_K, the kernel should handle that. The code uses masks and checks to avoid out-of-bounds access.

But in the kernel code, when loading the blocks, using boundary checks. For example, when loading a block from A, if the remaining K is less than BLOCK_SIZE_K, the mask would prevent accessing beyond the matrix.

Alternatively, the code uses block pointers with boundary checks automatically handled by Triton.

In the Triton kernel, when using tl.make_block_ptr, the shape parameter is (M, K) for A, and (K, N) for B. The block shape is (BLOCK_SIZE_M, BLOCK_SIZE_K) for A and (BLOCK_SIZE_K, BLOCK_SIZE_N) for B. The strides are (K, 1) for A (row-major) and (N, 1) for B (row-major), but wait, B is (K, N), so each row is of size N. So the strides for B would be (N, 1). However, when moving along the K dimension, the pointer for B would need to step in the first dimension (since B is K rows of N elements each). Wait, perhaps the block_ptr for B should be constructed with the correct strides.

Wait, block_ptr's strides are the strides of the tensor. For a tensor B of shape (K, N), in row-major layout, the strides are (N, 1). So when advancing along the K dimension (the first dimension), the pointer moves by N elements each step. But in the kernel, when loading a block from B, perhaps we need to transpose it to get the correct access pattern. Alternatively, perhaps the kernel is written to handle B as column-major, but that would require transposing B.

Hmm, this is a critical part. If the kernel is not handling the strides correctly, the matrix multiplication will be incorrect.

Let me think again. The standard approach in Triton for matrix multiplication is to tile both matrices into blocks and compute the product. For each block in C, which is (BLOCK_SIZE_M, BLOCK_SIZE_N), the computation involves the corresponding row block of A (BLOCK_SIZE_M x K) and column block of B (K x BLOCK_SIZE_N). But since K is the common dimension, we split K into chunks of BLOCK_SIZE_K, and compute the product in chunks.

So, for each iteration in K:

- Load a block of A of size BLOCK_SIZE_M x BLOCK_SIZE_K.
- Load a block of B of size BLOCK_SIZE_K x BLOCK_SIZE_N.
- Multiply these two blocks (resulting in BLOCK_SIZE_M x BLOCK_SIZE_N) and accumulate.

The key is that each block of A is a slice along the K dimension, and each block of B is a slice along the K dimension as well. Thus, for A, which is (M, K), each block is a tile of (BLOCK_SIZE_M, BLOCK_SIZE_K). For B, which is (K, N), each block is a tile of (BLOCK_SIZE_K, BLOCK_SIZE_N). But how to correctly access these blocks?

In Triton, using block pointers can help. For matrix A, the block pointer starts at (row_offset, 0) and moves along the K dimension. For matrix B, the block pointer starts at (0, column_offset) and moves along the K dimension. Wait, perhaps not. Let's see.

Block pointers for A:

- The base pointer is A.
- The shape is (M, K).
- The block shape is (BLOCK_SIZE_M, BLOCK_SIZE_K).
- The order might be (1, 0) to split M and K into blocks. Wait, the order parameter in make_block_ptr determines the order in which the block is traversed. For row-major access, the order is (0, 1). But I'm not entirely sure.

Alternatively, perhaps the block pointers are created such that for A, each block is a (BLOCK_SIZE_M, BLOCK_SIZE_K) tile starting at (m_offset, k_offset) and for B, each block is a (BLOCK_SIZE_K, BLOCK_SIZE_N) tile starting at (k_offset, n_offset).

This way, when k_offset increases by BLOCK_SIZE_K, the blocks move along the K dimension.

But to get this right, the code must correctly define the block pointers. Let's refer to Triton documentation or examples.

Looking at the Triton matrix multiplication example (like in the tutorials), the code for the kernel is structured as follows:

For a block in C (M, N), the code loops over K in steps of BLOCK_SIZE_K. For each step, it loads a block from A (BLOCK_SIZE_M, BLOCK_SIZE_K) and a block from B (BLOCK_SIZE_K, BLOCK_SIZE_N). These are multiplied and accumulated.

The block pointers for A and B are created with:

a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

But perhaps using block pointers simplifies this.

In the Triton example using block pointers, they are defined as:

For A:
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_SIZE_M, 0),
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
    order=(1, 0)
)

But I'm not sure. Alternatively, the order might determine how the block is laid out. This is crucial for correctness.

Alternatively, maybe using tl.dot is better. The tl.dot function can take two blocks and compute their product, handling the accumulation automatically. So, if the blocks are loaded correctly, the product is correct.

In the kernel code, after loading a_block and b_block, which are of size (BLOCK_SIZE_M, BLOCK_SIZE_K) and (BLOCK_SIZE_K, BLOCK_SIZE_N), respectively, the tl.dot(a_block, b_block) would compute a (BLOCK_SIZE_M, BLOCK_SIZE_N) block, which is accumulated into the accumulator.

So the code for the kernel would be something like:

@triton.autotune(...)
@triton.jit
def matmul_kernel(...):
    # compute offsets
    # create block pointers for A and B
    # loop over K
    # load a_block and b_block
    # compute product and accumulate
    # store result

Now, in terms of code structure, the kernel needs to have the correct parameters. The Triton kernel's parameters should include pointers to A, B, C, and the dimensions M, N, K. The strides for A, B, and C are also necessary.

The generated matmul_triton function will handle the device, data type, and tensor allocation. For example:

def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1)
    matmul_kernel[grid](A, B, C, M, N, K, ...)
    return C

But wait, the BLOCK_SIZE_M, etc., are determined by the autotuner, so the kernel's config is selected at runtime. Therefore, the grid calculation must use the current block size. But how is this handled when using autotune? Triton's autotuner automatically selects the best config based on the input sizes, so the grid is dynamic.

Alternatively, the grid is computed inside the kernel launch. Wait, no— the grid is specified when launching the kernel. But how to handle varying block sizes? For example, the block sizes are part of the autotune config. So the grid must be computed based on the block sizes for each config. However, Triton's autotuner handles this by compiling multiple versions of the kernel with different block sizes, and for each input size, the grid is computed for that block size.

Hmm, this part might require passing the block sizes as kernel parameters, but when using autotune, the config parameters (like BLOCK_SIZE_M) are part of the kernel's configuration.

In the Triton autotune example, the kernel's parameters include the configurable variables. For example:

configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
    ...
]

Then, in the kernel, these variables are used as static values. So the grid is computed as (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)), but how is this done in the wrapper function when the block sizes vary per config?

Ah, the grid is a function that takes the meta parameters. In the kernel launch, the grid is a lambda that takes the config's parameters.

Wait, in the Triton documentation, when using autotune, the grid function can be a callable that takes the kernel's parameters, including the config's params. For example:

def grid(META, A, B, C):
    return (triton.cdiv(A.shape[0], META['BLOCK_SIZE_M']), triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), 1)

Then, when the kernel is launched with matmul_kernel[grid](...), the grid is computed dynamically based on the META parameters (from the autotune config). This allows the grid to adapt to different block sizes for each autotune configuration.

So in the wrapper function matmul_triton, the grid is a function that uses the META parameters to compute the grid size.

Putting this together, the code inside matmul_triton would be:

def matmul_triton(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    # Check tensor dimensions and contiguity
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    # Wait, no. The grid should be 2D: (num_blocks_m, num_blocks_n)
    # So grid lambda would be (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    # But how to pass META to that?
    # The grid function in Triton can take a meta parameter which includes the current config's parameters.
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_SIZE_M']),
            triton.cdiv(N, meta['BLOCK_SIZE_N']),
        )
    matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))
    return C

But the kernel's parameters must include the strides. So in the kernel definition, parameters like stride_am (stride for A's M dimension, which is A.stride(0)), stride_ak (A's K dimension, which is A.stride(1)), and similarly for B and C.

So the kernel's parameters would be:

a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn

Now, in the kernel code, these strides are used to create the block pointers.

Putting all this together, the Triton kernel will use the strides to correctly navigate the matrices.

Another thing to consider is the data type. The input is float16, but the accumulation is done in float32. So when loading the blocks from A and B, they are cast to float32.

The kernel code would then look like:

a = tl.load(a_block_ptr, boundary_check=(0,1))
b = tl.load(b_block_ptr, boundary_check=(0,1))
a = a.to(tl.float32)
b = b.to(tl.float32)
c += tl.dot(a, b)

But wait, the tl.dot function automatically accumulates in float32 if the inputs are float16. So maybe casting is not needed, but to ensure precision, it's better to cast.

Now, testing for correctness: the verify_correctness function will compare the Triton output with the PyTorch output. The input_generator normalizes A and B, which might make the values more manageable and reduce numerical errors.

Potential issues with normalization: if A is normalized along dim=-1 (K), then each row of A has unit norm. Similarly, each row of B is normalized along dim=-1 (N), which would be the columns of B since B is (K, N). Wait, no. B is (K, N), so normalizing along dim=-1 (N) would normalize each row of B to have unit norm along the columns. Wait, the normalization is done as torch.nn.functional.normalize(A, dim=-1) for A (M, K). So each row (dimension 0) is a vector of length K, and normalization is done along the last dimension (dim=-1, K), which is correct. Each row of A is a unit vector. For B, same: normalize along dim=-1 (N), so each row of B is a unit vector along the columns. But matrix multiplication is sum over K, so if both A and B rows are unit vectors, the product's elements are cosines between A's rows and B's columns. This might help with numerical stability.

But the main point is that the Triton kernel and the PyTorch function compute the same product, so the verification should pass.

Another consideration is that the Triton kernel may have tiles that don't perfectly divide M, N, or K. Hence, the kernel needs to handle these cases with masking. The block pointers' boundary_check parameter can handle this.

In the triton.make_block_ptr calls, setting boundary_check=(0,1) would enable checks for the first and second dimensions. For example, for A's block pointer, if the block extends beyond M or K, the load would be masked. But I need to check the Triton documentation.

Alternatively, the kernel can handle out-of-bound accesses by checking the indices manually and using masks, but using block pointers with boundary_check is more efficient.

Putting it all together, the code for matmul_triton.py is as follows.
</think>

```python
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
```

<test>
```bash
python matmul_triton.py
```
</test>