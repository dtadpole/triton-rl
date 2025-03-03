API: deepseek
Test Case: ./func_softmax/
Timestamp: 20250303_053808

<think>
Okay, so I need to generate the Triton code for softmax. Let me start by recalling how softmax works. Softmax takes a vector of numbers and scales them so that the sum of the output is 1, with each element exponentiated and divided by the sum of exponents. The challenge in Triton is to do this efficiently, especially handling the max and sum reductions across rows.

First, the Triton kernel will process each row of the input tensor. For each row, I need to compute the maximum value to avoid numerical instability (subtracting the max), then compute the sum of exponentials, and finally divide each element by this sum.

In Triton, I should use blocks to process rows. Each program instance (block) handles a row. The input is a 2D tensor of shape (batch_size, dim_size). The kernel needs to load the row data, compute the max, compute the sum of exp(values - max), then compute each exp(value - max) / sum.

Wait, how to handle the reductions? For max, I can use a reduction across the row. Similarly for the sum. Triton has helper functions for this. Maybe using tl.max and tl.sum with the appropriate axis.

The steps would be:

1. Load the row data from the input pointer.
2. Compute the row's max value.
3. Subtract the max from each element to stabilize softmax.
4. Compute exp of each element.
5. Compute the sum of these exponents.
6. Divide each exponentiated element by the sum to get the softmax output.
7. Store the result back to the output.

Now, about the Triton kernel parameters. The rows are independent, so each block can handle a row. The grid should be (batch_size,), and each block has a number of threads that can handle the dim_size. But Triton uses block sizes that are powers of two, so maybe split the row into multiple blocks if necessary. Wait, Triton's programming model uses a single block per instance, but each instance can process a row. So the kernel would have a grid of (batch_size,) and each instance processes one row.

But how to handle different dim_sizes? The kernel should be able to handle any size, using a loop if necessary. For example, if the row is longer than the number of elements that can be processed in one iteration, use a loop with strides.

Let me sketch the kernel function. The input is a pointer to the tensor, and the output is another tensor. The kernel will have BLOCK_SIZE as a power of two, maybe 1024. Each row is processed by one block.

Wait, but for rows longer than BLOCK_SIZE, how to handle that? Oh right, we can loop over the row in chunks of BLOCK_SIZE elements. So each block loads a chunk, computes local max, then reduces across all chunks for the row's max. Similarly for the sum.

Wait, but softmax requires that all elements of a row are considered for the max and sum. So each row is processed by a single block. But if the row is too long, say 4096 elements, and BLOCK_SIZE is 1024, then each block would need to process 4 chunks. But that might not be efficient. Alternatively, Triton's grid can have multiple blocks per row, but that complicates the reduction. So perhaps the best way is to have each row handled by a single block, and process the row in chunks if necessary.

So, the steps in the Triton kernel:

Each program instance (block) handles one row. The block's ID is the row index. Then, for the elements in the row, load them in chunks, compute the max, subtract the max, exponentiate, sum, then normalize.

But how to compute the max of the entire row? Using a reduction across all elements. Triton's tl.max function can help here. Similarly, the sum of exponents.

So the code steps:

1. Get the row index from the program ID.
2. Compute pointers to the start of the row in the input and output tensors.
3. Initialize variables to track max and sum.
4. Loop over the elements of the row in chunks of BLOCK_SIZE.
   a. Load the current chunk from input.
   b. Compute the local max of this chunk, update the overall max.
5. After processing all chunks, the max for the row is known.
6. Loop over the elements again to compute exp(x - max) and sum them.
   a. For each chunk, subtract max, exponentiate, and accumulate the sum.
7. Then loop again to divide each exp(x - max) by the sum and store to output.

Wait, but that's three passes over the data. Maybe we can combine some steps. Alternatively, use two passes: first to compute the max, then a second pass to compute exp(x - max) and sum, then a third pass to normalize. But that's three passes. However, Triton's fused operations might make this efficient.

Alternatively, store the intermediate values in shared memory. Hmm, but Triton's model abstracts away some of the low-level details. Let's look up a Triton softmax example for guidance.

Wait, from what I remember, the standard approach is:

- Each row is processed by a block.
- In the first step, compute the row's max by iterating over elements and doing a reduction.
- Then, compute the sum of exp(x_i - max) in a similar reduction.
- Finally, compute each element's softmax as exp(x_i - max)/sum, and write to output.

But how to handle this in Triton. Let me outline the code.

The kernel function would take the input and output pointers, and the number of columns (dim_size). The kernel uses a block size (BLOCK_SIZE) that's a power of two. For each row:

row_idx = pid = tl.program_id(axis=0)
row_start = row_idx * dim_size
offs = row_start + tl.arange(0, BLOCK_SIZE)
mask = offs < row_start + dim_size

Then, first pass to compute max:

max_val = -infinity
for i in range(0, dim_size, BLOCK_SIZE):
    cols = i + tl.arange(0, BLOCK_SIZE)
    a = tl.load(input_ptr + cols, mask=cols < (row_start + dim_size), other=-float('inf'))
    current_max = tl.max(a, axis=0)
    max_val = tl.maximum(max_val, current_max)

Wait, maybe not. Alternatively, for each row, load all elements, compute max via a reduction.

Alternatively, perhaps using tl.reduce along the row. But how to handle rows longer than the block size.

Hmm, perhaps the correct way is to process the row in chunks, and for each chunk, compute the max, then perform a reduction across those max values. Wait, but each block processes one row, and each thread in the block processes a chunk. Wait, maybe not. Let me think again.

Wait, Triton's program model uses blocks (like CUDA), but each block is a group of threads. So for a row, perhaps the block is of size BLOCK_SIZE, and each thread in the block processes a chunk. So for a row of size N, we need to split it into chunks of BLOCK_SIZE elements. Then, each thread in the block loads a chunk, computes the max for that chunk, and then a reduction is done across all threads in the block to get the max of the entire row. Then, similarly for the sum of exponents.

This approach would require that the block's threads collectively process the row.

So steps in code:

1. Each block handles one row. The block's program ID (pid) is the row index.
2. The row has dim_size elements. The block has BLOCK_SIZE threads. So each thread processes (dim_size + BLOCK_SIZE - 1) // BLOCK_SIZE elements.
3. For the max computation: each thread loads its elements, computes the local max, then all threads in the block perform a reduction to find the global max for the row.
4. Once max is known, each thread computes exp(x_i - max) for its elements, sums them locally, then a block reduction for the total sum.
5. Finally, each thread divides the exp(x_i - max) by the sum and writes to output.

Alternatively, maybe using the Triton library functions for max and sum reductions. Let's look at an example.

From the Triton documentation, the softmax example uses the following approach:

- Each row is handled by a block.
- The block processes the row in chunks. For each chunk, compute the max and sum, then reduce.

But perhaps the code would look like this:

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets
    row_mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=row_mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(output_ptr + row_start + col_offsets, softmax_output, mask=row_mask)

But wait, this code assumes that the row fits into the BLOCK_SIZE. So if n_cols exceeds BLOCK_SIZE, this won't work. Therefore, the example code is only applicable when the row size is <= BLOCK_SIZE.

But the user's test cases include dim_sizes up to 4096. So if BLOCK_SIZE is 1024, then this approach would not handle 4096 elements. Therefore, the kernel needs to handle rows longer than BLOCK_SIZE.

Thus, the code must process the row in multiple steps. So the correct approach is to loop over the row in chunks of size BLOCK_SIZE.

Let me adjust the code to handle that.

For max reduction:

Compute the maximum value across all elements in the row. Each thread processes a chunk, computes the max of its chunk, then a reduction across all chunks in the row.

Similarly for sum.

So, here's the plan:

1. Each block handles a single row. The block's program ID is the row index.
2. The block's threads split the row into chunks of BLOCK_SIZE elements. For each chunk:
   a. Load the elements.
   b. Compute the local max of the chunk.
   c. Update the block's max.
3. After processing all chunks, compute the global max for the row.
4. Then, process each chunk again to compute the exp(x - max) and accumulate the sum.
5. After getting the sum, process each chunk again to compute the normalized value and store.

But how to handle this with Triton's programming model. Let's outline the code.

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    # input_row_stride is the number of elements to skip to get to the next row. For contiguous tensor, it's n_cols.
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Compute max of the row
    max_val = float('-inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        input_ptrs = input_ptr + row_start + col_idx
        curr_elements = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        curr_max = tl.max(curr_elements, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    
    # Compute sum(exp(row - max_val))
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        input_ptrs = input_ptr + row_start + col_idx
        curr_elements = tl.load(input_ptrs, mask=mask, other=0.0)
        curr_elements_minus_max = curr_elements - max_val
        exp_values = tl.exp(curr_elements_minus_max)
        sum_exp += tl.sum(exp_values, axis=0)
    
    # Compute softmax and write to output
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        input_ptrs = input_ptr + row_start + col_idx
        output_ptrs = output_ptr + row_start + col_idx
        curr_elements = tl.load(input_ptrs, mask=mask, other=0.0)
        curr_elements_minus_max = curr_elements - max_val
        exp_values = tl.exp(curr_elements_minus_max)
        softmax_values = exp_values / sum_exp
        tl.store(output_ptrs, softmax_values, mask=mask)

Wait, but in this code, each of the loops (for max, sum, and store) process the row in BLOCK_SIZE chunks. Each thread in the block processes a chunk? Or is each program instance handling a row with a single thread? No, because in Triton, the program_id(0) gives the block index. So each block handles a row. But the block can have multiple threads. Wait, no. Wait, Triton's programming model is a bit different. Each "program instance" is a block, and the grid is set up such that each row is processed by one program instance. The block's threads are used for vectorized operations. So, in the code above, the for loops are processing the row in chunks, and within each chunk, the entire block is used to load the data. For example, BLOCK_SIZE is 1024, each chunk is 1024 elements, processed by 1024 threads. But if the row is 4096 elements, then the loop runs 4 times. Each time, the block processes a chunk of 1024 elements. So each iteration is a separate load and computation.

But how does the reduction work for max_val and sum_exp? Because each iteration processes a chunk, the code computes the max of each chunk, and keeps updating the max_val with the maximum between the current chunk's max and the previous max. Similarly for the sum_exp.

But this approach is correct. Because the max of the entire row is the maximum of all chunks' max values. Similarly, the sum is the sum of all chunks' sum of exp.

Once max and sum are computed, the final loop loads each chunk again, computes the exp(x - max), divides by sum, and stores.

This should handle rows of any size, as long as the kernel is called with a grid of size (batch_size,) and BLOCK_SIZE is a divisor of the row length or larger than the row length.

Now, the code above has three loops over the row. Each loop processes the row in chunks. This could be optimized, but for correctness, this should work.

Now, for the Triton function, we need to create a helper function that calls this kernel. The helper function would handle the tensor dimensions and launch configuration.

So, the Python function would look like:

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(...):
    ...

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    batch_size, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)  # Adjust based on hardware limits
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel
    grid = (batch_size,)
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

Wait, but the next_power_of_2 may be larger than the actual n_cols. For example, if n_cols is 1025, next_power_of_2 is 2048. But then, the BLOCK_SIZE should be set to the maximum possible allowed. Wait, but in the kernel, the for loop handles the chunks, so even if BLOCK_SIZE is larger than n_cols, the first iteration (col_offset=0) would process the entire row. For example, if n_cols=64 and BLOCK_SIZE=128, the loop will process col_offset=0, col_idx ranges from 0 to 127, but the mask would be true for 0-63. So the code is still correct.

But setting BLOCK_SIZE to the next power of two of the row size may not be optimal. Alternatively, set BLOCK_SIZE to 1024, which is a common choice. Because if the row is longer than 1024, the for loop will handle the chunks. But how to choose BLOCK_SIZE optimally? Perhaps using a fixed block size, like 1024, and let the loop handle any row size.

But in the code above, the BLOCK_SIZE is passed as a kernel argument. In the helper function, perhaps we can set BLOCK_SIZE to 1024, but for small rows, it's better to use a smaller block size. However, Triton allows for the BLOCK_SIZE to be a compile-time constant, so the kernel must be compiled with different BLOCK_SIZE values for different row sizes. That's not practical. Therefore, a better approach is to choose a fixed BLOCK_SIZE (like 1024) that works for all row sizes, and let the kernel loop through the row in chunks of that size.

So, in the helper function, set BLOCK_SIZE=1024, regardless of the row size. Then, for any row size, the kernel loops in steps of 1024 elements. So the code becomes:

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    # The kernel is compiled for a specific BLOCK_SIZE, so we must choose a fixed size here.
    # Alternatively, use autotune to select the best BLOCK_SIZE.
    # For simplicity, set BLOCK_SIZE to the next power of two up to 1024.
    BLOCK_SIZE = 1024  # Fixed block size.
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

But this may not be optimal. However, given the time constraints, perhaps proceed with this approach.

Now, the user also requires a generated_torch_func for testing. So, write a PyTorch softmax function for comparison.

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

Then, in the main code, call the benchmark functions as per the user's code.

Putting it all together, the code for softmax_triton.py will include:

- The Triton kernel function with proper documentation.
- The Python functions triton_softmax and torch_softmax.
- The main block that runs the tests and benchmarks.

Now, for the documentation comments as per the user's requirement.

The Triton kernel's docstring should explain:

Description: Computes the softmax of each row in the input tensor using Triton.
Args:
    output_ptr (triton.PointerType): Pointer to the output tensor.
    input_ptr (triton.PointerType): Pointer to the input tensor.
    input_row_stride (int): Stride in elements between consecutive rows in the input.
    output_row_stride (int): Stride in elements between consecutive rows in the output.
    n_cols (int): Number of columns (elements per row).
    BLOCK_SIZE (tl.constexpr): Block size for processing elements in chunks.

Explanation steps:
1. Determine the row index from the program ID.
2. Compute the maximum value of the row by iterating over elements in chunks.
3. Compute the sum of exponentials of (each element minus the maximum).
4. Normalize each element by the sum of exponentials and store the result.

Similarly, the Python functions (triton_softmax and torch_softmax) need their docstrings.

But the user's instruction says to generate the triton code for softmax (filename: softmax_triton.py), which includes the Triton kernel and the helper functions. And the main code should test correctness and performance.

Now, putting it all together.

Potential issues:

- Handling of BLOCK_SIZE: The Triton kernel's loop must correctly process all elements. For example, when n_cols is not a multiple of BLOCK_SIZE, the mask should be applied properly.
- Strides: For non-contiguous tensors, the input and output strides must be handled. But in the test code, the input is generated with torch.randn, which is contiguous. So assuming contiguous tensors may be okay, but the kernel is written to handle arbitrary strides.
- Numerical stability: Subtracting the max before exponentiating.
- Performance: The Triton code should be efficient, but with three passes over the data (max, sum, store), it might be slower than a hand-optimized kernel. But for the purpose of this question, perhaps this is acceptable.

Testing the code with the given main block:

The main block calls verify_correctness_func and benchmark_performance_func from benchmark.py. The input generator creates a tensor x of shape (batch_size, dim_size). The Triton function and Torch function are compared.

So the code in softmax_triton.py would look like:

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes the softmax activation for each row of the input tensor.

    Args:
        output_ptr (triton.PointerType): Pointer to the output tensor. Shape (batch_size, n_cols)
        input_ptr (triton.PointerType): Pointer to the input tensor. Shape (batch_size, n_cols)
        input_row_stride (int): Stride in elements between consecutive rows in the input tensor.
        output_row_stride (int): Stride in elements between consecutive rows in the output tensor.
        n_cols (int): Number of columns (elements per row).
        BLOCK_SIZE (tl.constexpr): Block size for processing elements in chunks.

    Returns:
        None: The output tensor is modified in-place.

    Explanation:
        1. Determine the current row index from the program ID.
        2. Iterate over row elements in chunks of BLOCK_SIZE to compute the row-wise maximum value.
        3. Iterate over elements again to compute the sum of exponentials (after subtracting max for numerical stability).
        4. Final iteration normalizes each element by the sum of exponentials and stores the result.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    output_row_start = row_idx * output_row_stride

    # Step 1: Compute row maximum
    max_val = float('-inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_start + col_offsets
        curr_row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        curr_max = tl.max(curr_row, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # Step 2: Compute sum of exponentials
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_start + col_offsets
        curr_row = tl.load(input_ptrs, mask=mask, other=0.0)
        exp_row = tl.exp(curr_row - max_val)
        sum_exp += tl.sum(exp_row, axis=0)

    # Step 3: Compute and store softmax
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_start + col_offsets
        output_ptrs = output_ptr + output_row_start + col_offsets
        curr_row = tl.load(input_ptrs, mask=mask, other=0.0)
        softmax_values = tl.exp(curr_row - max_val) / sum_exp
        tl.store(output_ptrs, softmax_values, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the softmax function row-wise using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor with softmax applied to each row. Shape (batch_size, dim_size).

    Explanation:
        1. Determines the number of rows and columns in the input tensor.
        2. Allocates output tensor with the same shape as input.
        3. Launches the Triton kernel with appropriate grid and block dimensions.
    """
    batch_size, n_cols = x.shape
    BLOCK_SIZE = 1024  # Fixed block size for processing elements
    y = torch.empty_like(x)
    grid = (batch_size,)
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the softmax function row-wise using PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor with softmax applied to each row. Shape (batch_size, dim_size).

    Explanation:
        1. Uses torch.softmax with dim=-1 to apply row-wise softmax.
    """
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])

Now, verifying the code:

- The Triton kernel's docstring follows the user's structure.
- The functions triton_softmax and torch_softmax have their docstrings.
- The main block uses the exact code provided by the user.

Potential issues:

- The Triton kernel uses output_row_start correctly, considering the output pointer's strides. In case the output tensor is non-contiguous, but in the test case, the output is allocated as empty_like(x), which is contiguous. So the code should work.
- The BLOCK_SIZE is fixed at 1024. For rows smaller than 1024, the first loop iteration will handle all elements. For rows larger than 1024, the loop will process chunks of 1024 elements. This should handle all cases.
- The verification function compares the Triton and Torch implementations across various batch and dim sizes.
</think>

```python
# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes row-wise softmax activation for a 2D tensor using Triton.

    Args:
        output_ptr (triton.PointerType): Output tensor pointer. Shape (batch, n_cols)
        input_ptr (triton.PointerType): Input tensor pointer. Shape (batch, n_cols)
        input_row_stride (int): Input tensor row stride in elements
        output_row_stride (int): Output tensor row stride in elements
        n_cols (int): Number of columns per row
        BLOCK_SIZE (tl.constexpr): Processing block size (elements per iteration)

    Returns:
        None: Output is stored through output_ptr

    Explanation:
        1. Calculate row index from program ID
        2. Find maximum value in row through block-reduce
        3. Compute sum of exp(x_i - max) for normalization
        4. Store normalized exp(x_i - max)/sum to output
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    output_row_start = row_idx * output_row_stride

    # Find row maximum
    max_val = float('-inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Compute normalization sum
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Write final normalized values
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0)
        exp_vals = tl.exp(vals - max_val)
        softmax_out = exp_vals / sum_exp
        tl.store(output_ptr + output_row_start + cols, softmax_out, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied row-wise. Shape matches input.

    Explanation:
        1. Validate input dimensions
        2. Configure kernel launch parameters
        3. Execute Triton kernel with automatic memory management
    """
    batch_size, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    y = torch.empty_like(x)
    grid = (batch_size,)
    
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax using PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied row-wise. Shape matches input.

    Explanation:
        1. Uses PyTorch's built-in softmax implementation
    """
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
```

<test>
To verify correctness and benchmark performance, run:

```bash
python softmax_triton.py
```
</test>