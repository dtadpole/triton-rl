API: deepseek
Test Case: ./func_rmsnorm/
Timestamp: 20250303_053536

<think>
Okay, I need to generate Triton code for RMSNorm in a file called rmsnorm_triton.py. The user also wants to test correctness and performance using the provided code. Let me start by recalling what RMSNorm does. RMSNorm is a normalization technique where each element is divided by the root mean square of the layer, scaled by a learned weight. 

First, I'll need to write the Triton kernel. The kernel should compute the RMS for each row, then normalize the input using that RMS value and the weight vector. Let's outline the steps:

For the Triton function:
1. Load the input data and weight from global memory.
2. Compute the sum of squares for each element in the row.
3. Use the sum to calculate the RMS value, adding epsilon for numerical stability.
4. Normalize the input by dividing by RMS, then multiply by the weight.

I should make sure that the kernel is properly parallelized over rows. Each program instance (block?) handles a row. Using Triton's pointers and block indices correctly is crucial. Also, using Triton's functions for efficient memory access, like tl.load and tl.store.

Next, the PyTorch version. That's straightforward using standard operations: calculate the variance along the last dimension, add epsilon, normalize, then multiply by weight.

Then, integrating the test code. The input generator creates random tensors for x and weight, along with a fixed epsilon. The verify_correctness_func and benchmark_performance_func from benchmark.py will compare the Triton and Torch implementations across different batch and dimension sizes.

Wait, the code the user provided uses generated_triton_func and generated_torch_func. So in the test section, I need to make sure those functions are correctly named. The Triton function is probably the compiled one. Oh right, Triton requires JIT compilation, so the Triton function should be decorated with @triton.jit, and then a wrapper function is created to handle the kernel launch.

So in the code, the Triton kernel will be a function with @triton.jit. Then, a Python function (like rmsnorm_triton) will configure the grid, launch the kernel, and return the output tensor.

Let me structure the code:

- Import necessary modules: torch, triton, etc.
- Define the Triton kernel with proper parameters.
- Create a wrapper function that calls the kernel with the right configuration.
- Define the PyTorch version as a function.
- Then, the code that uses the input generator and calls the benchmark functions.

Let me check the parameters for the kernel. The x and weight pointers should be of type tl.tensor, and the stride information is important. The kernel should have BLOCK_SIZE as a configurable parameter. Also, the eps is a scalar.

Wait, in the Triton kernel, how do I handle different rows? Each row is processed in parallel. So the program ID along the first axis (axis=0) will correspond to the row index. Then, for each row, the kernel processes the elements in that row.

For coalesced memory access, the BLOCK_SIZE should be a power of two and as large as possible, but within Triton's constraints (like 1024). Maybe set it as the next power of two less than the dimension size. But the kernel should handle varying BLOCK_SIZE via a config parameter.

In the kernel launch configuration, the grid should be (batch_size,) for rows, and the block is (BLOCK_SIZE,). Wait, no, in Triton, the grid is the number of blocks, and each block has a size. Wait, maybe the kernel uses a 1D grid where each block processes a row. But considering that each row's elements are processed in parallel, the BLOCK_SIZE should be sufficient to cover the row's dimension. So for dim_size up to 4096, the BLOCK_SIZE should be a multiple that covers it. For example, if the row is 4096 elements, and BLOCK_SIZE is 1024, then each block would process 1024 elements, but we'd need 4 blocks per row. Wait, maybe I'm mixing up the grid and block structure here.

Alternatively, for each row, the kernel uses a single block, processing all elements in the row. That would require that the BLOCK_SIZE is at least the row size. But if the row is larger than the maximum allowed BLOCK_SIZE (like 1024), that's a problem. So perhaps we need to split the row processing into multiple blocks. However, for simplicity and efficiency, maybe we assume that the row size is a multiple of BLOCK_SIZE. But the user's test cases include dim_sizes up to 4096. So the BLOCK_SIZE needs to be a divisor of all dim_sizes. Wait, but the input generator uses varying dim_sizes. So the kernel needs to handle any dim_size by using a BLOCK_SIZE that can be adjusted. Alternatively, the kernel could dynamically adjust based on the actual dim_size.

Wait, perhaps using the num_warps parameter. For example, setting BLOCK_SIZE as the next power of two that's greater than or equal to the dim_size. But that's not possible if the dim_size is larger than the maximum allowed (like 4096). Alternatively, maybe the kernel uses a for loop with a range based on the row size and the block size. For example, each thread in the block handles multiple elements via a loop. That way, even if the row size is larger than the block size, the kernel can process all elements.

So the kernel code for RMSNorm would:

1. Get the row index via program ID.
2. Compute the offset for this row's start in the input tensor.
3. Compute the sum of squares for the elements in the row.
   a. Initialize a sum variable.
   b. Loop over the elements in the row, in increments of BLOCK_SIZE.
   c. For each element, load it, square it, accumulate.
4. Compute the RMS (sqrt(sum / N + eps)).
5. Then, loop again over the elements to compute the normalized value and multiply by the weight, storing the result.

Wait, but doing two passes over the data (one for sum, then another for normalization) could be inefficient. However, for numerical stability and correctness, we need the RMS value for the entire row before normalizing each element. So two passes are necessary.

In Triton, how to handle the sum across the elements? Since each row is processed by a single block, we can use block-level reduction. For example, each thread sums its portion, then a block-level reduction is done to get the total sum for the row.

Wait, but in this case, each block is handling a row. So the block's threads process the elements of the row. For the sum, each thread can accumulate the sum of its elements, then a block-level reduction (like using tl.sum) across all threads in the block to get the total sum for the row. Then, compute RMS once per row. Then, each thread processes its elements again to compute the normalized value.

So the steps would be:

For each row (each program ID is a row):
- Compute the sum of squares for all elements in the row.
   - Use a loop with step size BLOCK_SIZE to cover the entire row.
   - Each thread loads an element, squares it, and accumulates.
   - Sum across all threads in the block to get the total sum for the row.
- Compute RMS.
- Loop again over the elements, normalize each by RMS, multiply by weight, and store.

But how to handle variable row lengths? The row length is the dim_size. The kernel needs to know the dim_size to determine how many elements to process. So in the kernel's parameters, we should pass the dim_size as a parameter. So the kernel will have an argument like 'dim_size: tl.constexpr'.

Also, the BLOCK_SIZE is a tl.constexpr parameter. So when launching the kernel, the BLOCK_SIZE is set to the next power of two less than or equal to dim_size. For example, if dim_size is 512, BLOCK_SIZE could be 512. If dim_size is 4096, maybe 1024. But then, the loop would iterate over (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE steps.

Wait, but the BLOCK_SIZE should be a power of two for efficient reductions. So the code should adjust the BLOCK_SIZE based on the dim_size.

But the user's input generator allows variable dim_sizes. How to handle that in the kernel? Since in Triton, the kernel is compiled for specific values of the tl.constexpr parameters. Therefore, to handle varying dim_sizes, the BLOCK_SIZE must be a parameter that can vary. Alternatively, the kernel could be compiled for each possible BLOCK_SIZE. But that's not practical. So maybe the approach is to split the row into chunks processed by the block, with each thread handling multiple elements.

Alternatively, the kernel is written in a way that the BLOCK_SIZE is a parameter, and when launching the kernel, we choose the largest possible BLOCK_SIZE that is a power of two and divides the dim_size. However, this may not be feasible for all dim_sizes. For example, if dim_size is 512, BLOCK_SIZE can be 512, 256, etc. But if the dim_size is 4096, perhaps we use a BLOCK_SIZE of 1024, and loop over 4 steps.

Wait, perhaps the kernel can loop over all elements in the row, regardless of BLOCK_SIZE. For example, each thread processes multiple elements. Let's consider that the row is processed in a loop where each thread handles multiple elements. The total number of elements is dim_size. The block's threads (number of threads is BLOCK_SIZE) will process all elements by iterating over the range.

So for a row of size N, each thread in the block (with index i) will handle elements i, i + BLOCK_SIZE, i + 2*BLOCK_SIZE, etc., up to N.

This way, even if BLOCK_SIZE is smaller than N, all elements are covered. This is a common approach in CUDA and Triton code for handling varying sizes.

So in the kernel:

For the sum step:

row_start = pid * stride (if the input is contiguous, the stride would be dim_size)
sum_sq = 0.0
for idx in range(0, dim_size, BLOCK_SIZE):
    offset = idx + tl.program_id(1)  # Wait, no. Maybe the program ID is the row index, and the block's threads process elements in the row.

Wait, perhaps the kernel is launched with a grid of (batch_size,), and each block handles one row. Then, within the block, each thread processes multiple elements of the row.

But in Triton, each program is a block of threads, and the grid is the number of blocks. So for a 1D grid with batch_size blocks, each block processes one row. Each block has BLOCK_SIZE threads.

Then, for each element in the row:

for i in range(0, dim_size, BLOCK_SIZE):
    elem_idx = i + thread_idx
    if elem_idx < dim_size:
        x = tl.load(x_ptr + elem_idx)
        sum_sq += x * x

Then, sum across all threads in the block. That can be done using tl.sum across the block.

Wait, but that's per-row processing. So each block processes one row, and each thread in the block processes multiple elements (if dim_size > BLOCK_SIZE).

Wait, but in this case, the block's threads would process all elements in the row by looping over the elements in steps of BLOCK_SIZE. Each thread accumulates partial sums. Then, a block-level reduction gives the total sum for the row.

That's the way to do it. Then, the RMS is computed once per row, and then another loop over the elements to compute the normalized value.

So the Triton kernel code would look like this:

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    dim_size,
    eps,
    stride_x_row,
    stride_weight,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    output_row_ptr = output_ptr + row_idx * stride_output_row
    weight_row_ptr = weight_ptr  # Assuming weight is 1D

    # Compute sum of squares
    sum_sq = 0.0
    for offset in range(0, dim_size, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < dim_size
        x = tl.load(x_row_ptr + col_idx, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    # RMS calculation
    rms = tl.sqrt(sum_sq / dim_size + eps)

    # Normalize and weight
    for offset in range(0, dim_size, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < dim_size
        x = tl.load(x_row_ptr + col_idx, mask=mask, other=0.0)
        weight_val = tl.load(weight_row_ptr + col_idx, mask=mask, other=0.0)
        output = x / rms * weight_val
        tl.store(output_row_ptr + col_idx, output, mask=mask)

Wait, but the sum_sq computed in the first loop is the sum over all elements in the row. Then, the rms is computed as sqrt(sum_sq / dim_size + eps). Then, each element is divided by rms and multiplied by the corresponding weight element.

But in the code above, each thread is summing x*x for each element, and then sum_sq is summed across all threads in the block. Wait, no. The sum_sq is accumulated per-thread, then summed across the block. Wait, no. The code uses tl.sum(x * x, axis=0), which sums over the entire vector (since x is a vector of BLOCK_SIZE elements). Then, sum_sq is added across the loop iterations, which would be summing all the x² values for the row. Wait, but the loop is over the row in chunks of BLOCK_SIZE. For each chunk, x is loaded (BLOCK_SIZE elements), each element squared, summed, and added to sum_sq. So sum_sq is the total sum of squares for the entire row.

Then, dividing by dim_size, adding eps, taking the square root gives RMS. Then, each element is divided by RMS and multiplied by the corresponding weight element.

That seems correct. But in Triton, when you do tl.sum on a tensor, it's a vector operation. So x * x is a vector of BLOCK_SIZE elements. tl.sum(x * x, axis=0) would sum all elements, resulting in a scalar. Then, sum_sq accumulates across loop iterations. So the sum_sq variable holds the sum of squares for all elements in the row.

Then, after computing rms, the second loop processes each element in the row again, divides by rms, multiplies by the weight, and stores to output.

Now, the Triton kernel needs to be called with the correct grid and block sizes. The grid is (batch_size,), and each block handles a row. The BLOCK_SIZE is a tunable parameter. But how to choose it? For the input_generator, since the dim_sizes vary, we need to set BLOCK_SIZE dynamically. However, in Triton, the BLOCK_SIZE must be a compile-time constant for the kernel. Therefore, for each call to the kernel, we need to choose the largest possible BLOCK_SIZE that is a power of two and less than or equal to the dim_size. If the dim_size isn't a power of two, that could be a problem. But for the test cases provided (64, 128, 256, 512, 1024, 2048, 4096), all are powers of two. So BLOCK_SIZE can be set to the dim_size each time. For example, if the dim_size is 4096, set BLOCK_SIZE to 4096. But Triton's maximum block size is 1024 (I think), so this may not be possible. Wait, Triton's block size (number of threads per block) is limited. For example, for the latest version, the maximum is 1024. So for dim_size 4096, BLOCK_SIZE can't be 4096. So the code would need to use a smaller BLOCK_SIZE, like 1024, and loop over 4 iterations for each row.

So, in the code that launches the kernel, the BLOCK_SIZE is set to the minimum between the maximum allowed (like 1024) and the dim_size, but ensuring it's a power of two. Alternatively, we can set BLOCK_SIZE to 1024 and handle all dim_sizes via loops.

Wait, but for dim_sizes larger than BLOCK_SIZE, the loop in the kernel will handle them. For example, if BLOCK_SIZE is 1024 and dim_size is 4096, the loop runs 4 times (0, 1024, 2048, 3072). Each iteration covers 1024 elements. But col_idx = offset + tl.arange(0, BLOCK_SIZE), so for offset=3072, col_idx ranges from 3072 to 4095 (since 3072+1024=4096, which is the dim_size). The mask ensures that col_idx < dim_size, which handles the case when dim_size is not a multiple of BLOCK_SIZE.

So the code should work for any dim_size, as long as BLOCK_SIZE is a power of two, which can be set to the maximum possible (like 1024). So in the kernel launch, the BLOCK_SIZE is set to 1024.

But then, for small dim_sizes (like 64), the loop runs once (since 64 <= 1024). So this approach works.

Therefore, in the Triton kernel launch, the BLOCK_SIZE is fixed to 1024. If the dim_size is larger than 1024, the loop will handle it. But for the test cases provided, the maximum dim_size is 4096, which requires 4 iterations (4096 / 1024 = 4).

Now, the wrapper function for the Triton kernel:

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_cuda and weight.is_cuda
    assert x.shape[-1] == weight.shape[-1], "x and weight must have the same last dimension"
    batch_size, dim_size = x.shape
    # Define kernel grid and BLOCK_SIZE
    BLOCK_SIZE = 1024  # or determine based on dim_size, but Triton requires constexpr
    grid = (batch_size,)
    rmsnorm_kernel[grid](x, weight, output, dim_size, eps, 
                        x.stride(0), weight.stride(0), output.stride(0),
                        BLOCK_SIZE=BLOCK_SIZE)
    return output

Wait, but the BLOCK_SIZE is set to 1024 regardless of dim_size. For example, if dim_size is 64, then BLOCK_SIZE=1024 is larger than needed, but the mask in the kernel's loops (col_idx < dim_size) will handle it. The for loops will run once, since 0 + 1024 > 64. So in the first iteration, offset=0, and col_idx ranges from 0 to 1023, but mask will ignore those beyond 63.

So that's okay.

Now, the PyTorch implementation:

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Compute RMS
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    # Normalize and scale
    return x / rms * weight

But wait, the weight is a vector of size dim_size. So after normalization (x / rms is [batch, dim]), multiplying by weight [dim] should be element-wise. So the code is correct.

Now, the test code:

batch_sizes = [256, 512, 1024, 2048];
dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    weight = torch.randn(dim_size, device=device, dtype=dtype)
    eps = 1e-5
    return x, weight, eps
from benchmark import verify_correctness_func, benchmark_performance_func
verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes]);
benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes]);

But in Python, the semicolons are not needed. Also, the lists for batch_sizes and dim_sizes should be passed as separate arguments. Wait, the input_generator takes batch_size and dim_size as parameters. The verify_correctness_func is called with [batch_sizes, dim_sizes], but perhaps the code should generate all combinations. Or maybe the code in benchmark.py expects to generate all possible combinations. I think the code provided by the user is correct as is.

Now, putting it all together in the file.

So the final code in rmsnorm_triton.py will have:

- The Triton kernel with proper documentation.
- The wrapper function.
- The PyTorch function.
- The test code.

Wait, but the user wants to generate the code that includes the test code. However, the test code uses the functions from benchmark.py, which is assumed to exist. So the code in rmsnorm_triton.py should include the test code as well. But usually, such code is placed inside a if __name__ == "__main__": block.

But the user's instruction says to "test correctness and performance using following code literally". So the code provided by the user should be included in the generated file.

So the generated file will:

1. Import necessary modules (torch, triton, etc.)
2. Define the Triton kernel with documentation.
3. Define the wrapper function (rmsnorm_triton).
4. Define the torch function (rmsnorm_torch).
5. Then, the code to run the tests and benchmarks as in the user's code.

But the code provided by the user has:

verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);

So the generated code should replace generated_triton_func and generated_torch_func with rmsnorm_triton and rmsnorm_torch.

So the code in the generated file will have all the necessary parts.

Now, let's check the documentation for each function. The user requested detailed comments for each function.

For the Triton kernel:

Description: Computes RMS normalization for input tensor using Triton.
Args:
    x_ptr (tl.tensor): Input tensor of shape (batch_size, dim_size).
    weight_ptr (tl.tensor): Weight tensor of shape (dim_size).
    output_ptr (tl.tensor): Output tensor of shape (batch_size, dim_size).
    dim_size (int): Dimension size of the input and weight tensors.
    eps (float): Small value added to the variance to avoid division by zero.
    ... (other strides)
    BLOCK_SIZE (tl.constexpr): Block size for Triton kernel execution.

Returns: None (output is stored in output_ptr).

Explanation:
    1. Each block processes a single row of the input tensor.
    2. Sum the squares of all elements in the row.
    3. Compute RMS value with epsilon.
    4. Normalize each element and multiply by weight, storing result in output.

For the rmsnorm_triton function:

Description: Applies RMS normalization using a Triton kernel.
Args:
    x (Tensor): Input tensor of shape (batch_size, dim_size).
    weight (Tensor): Weight tensor of shape (dim_size).
    eps (float): Epsilon for numerical stability.

Returns:
    Tensor: Normalized tensor of same shape as x.

Explanation:
    1. Allocate output tensor.
    2. Launch Triton kernel with appropriate grid and block configuration.
    3. Return output tensor.

For the rmsnorm_torch function:

Description: PyTorch implementation of RMS normalization.
Args and Returns similar to Triton function.

Now, putting all together in code with documentation.

Final checks:

- Ensure the Triton kernel and functions have proper type hints and docstrings.
- The code uses the correct parameters and variable names.
- The test code is included and calls the correct functions.

Possible issues to consider:

- Strides: The Triton kernel uses the x's row stride. For a contiguous tensor, x.stride(0) is dim_size, but need to confirm.

For example, x = torch.randn(batch_size, dim_size) is contiguous in C-order, so x.stride(0) is dim_size, and x.stride(1) is 1. So in the kernel, x_row_ptr is x_ptr + row_idx * stride_x_row (which is dim_size * row_idx). Then, each element in the row is accessed with x_row_ptr + col_idx. Assuming the row is contiguous. But if the input tensor is not contiguous, this may not work. However, the input_generator uses x = torch.randn(batch_size, dim_size, device=device, dtype=dtype), which is contiguous. So the code is okay.

Another possible issue: the weight tensor in Triton is 1D, so weight_row_ptr is weight_ptr, and each element is accessed with col_idx. So for each element in the row, the weight is at the same col_idx. That's correct.

Now, the code should handle different data types (like float16) as per the input_generator.

Testing:

The test code should be placed inside a main guard. So the code at the end of the file would be:

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])

But note that the verify_correctness_func and benchmark_performance_func are passed [batch_sizes, dim_sizes], which I assume means they will test all combinations of batch and dim sizes. The benchmark.py functions should handle that.

Finally, the test command would be to run the script:

python rmsnorm_triton.py

But the user may need to ensure that CUDA is available and Triton is installed.
</think>

```python
# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    dim_size,
    eps,
    stride_x_row,
    stride_weight,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes RMS normalization for input tensor using Triton.

    Args:
        x_ptr (tl.tensor): Input tensor of shape (batch_size, dim_size).
        weight_ptr (tl.tensor): Weight tensor of shape (dim_size).
        output_ptr (tl.tensor): Output tensor of shape (batch_size, dim_size).
        dim_size (int): Dimension size of the input and weight tensors.
        eps (float): Small value added to variance for numerical stability.
        stride_x_row (int): Stride between rows in input tensor.
        stride_weight (int): Stride for weight tensor (should be 1).
        stride_output_row (int): Stride between rows in output tensor.
        BLOCK_SIZE (tl.constexpr): Triton block size (must be power of two).

    Returns:
        None: Output is stored in output_ptr.

    Explanation:
        1. Each block processes one row from the input tensor.
        2. First loop computes sum of squares of elements in row.
        3. Calculate RMS using sum, dim_size and epsilon.
        4. Second loop normalizes elements and applies weight.
    """
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    output_row_ptr = output_ptr + row_idx * stride_output_row

    # Compute sum of squares
    sum_sq = 0.0
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    rms = tl.sqrt(sum_sq / dim_size + eps)

    # Apply normalization and weights
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        normalized = (x / rms) * w
        tl.store(output_row_ptr + cols, normalized, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS normalization using Triton kernel.

    Args:
        x (Tensor): Input tensor of shape (batch_size, dim_size).
        weight (Tensor): Weight tensor of shape (dim_size).
        eps (float): Epsilon for numerical stability.

    Returns:
        Tensor: Normalized output tensor of same shape as input.

    Explanation:
        1. Verifies input dimensions match weight shape.
        2. Initializes output tensor.
        3. Configures kernel launch parameters.
        4. Launches Triton kernel with optimized block size.
    """
    output = torch.empty_like(x)
    batch_size, dim_size = x.shape
    
    # Adjust BLOCK_SIZE to largest power-of-two <= dim_size
    max_block_size = 1024  # Triton's maximum recommended block size
    BLOCK_SIZE = min(max_block_size, triton.next_power_of_2(dim_size))
    
    grid = (batch_size,)
    rmsnorm_kernel[grid](
        x, weight, output,
        dim_size,
        eps,
        x.stride(0),
        weight.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch implementation of RMS normalization.

    Args:
        x (Tensor): Input tensor of shape (batch_size, dim_size).
        weight (Tensor): Weight tensor of shape (dim_size).
        eps (float): Epsilon for numerical stability.

    Returns:
        Tensor: Normalized output tensor of same shape as input.

    Explanation:
        1. Compute mean of squared values along last dimension.
        2. Add epsilon and compute reciprocal square root.
        3. Multiply normalized values by weight element-wise.
    """
    rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x * (weight / (rms + eps))

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
```

<test>
To test the RMSNorm implementation with correctness verification and performance benchmarking:

```bash
python rmsnorm_triton.py
```

Note: Ensure PyTorch with CUDA support and Triton are properly installed in your environment before running.
</test>