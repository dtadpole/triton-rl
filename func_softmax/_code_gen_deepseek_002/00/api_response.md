API: deepseek
Test Case: ./func_softmax/
Timestamp: 20250303_003524

<think>
Alright, I need to generate Triton code for a softmax function and test it using the provided code. Let's start by recalling how softmax works. Softmax applies the exponential function to each element and normalizes by the sum of exponentials across the dimension. The challenge is to implement this efficiently in Triton.

First, the Triton function should handle 1D softmax along the last dimension. The input is a 2D tensor (batch_size, dim_size). The Triton kernel needs to compute the maximum and sum of exponentials for each row to avoid numerical instability.

I remember that in Triton, we can use reduction operations. For each row, the steps are:
1. Load the elements into a block.
2. Compute the max value in the row.
3. Subtract the max from each element and exponentiate.
4. Compute the sum of these exponentials.
5. Divide each element by the sum.

But how to structure the Triton kernel? Let's think about the kernel parameters. The rows can be processed in parallel. Each program (block) handles a row. The number of columns (dim_size) might be larger than the block size, so we need to process elements in chunks.

The Triton kernel will have a loop over the columns. For each row, we'll iterate through the columns in steps of BLOCK_SIZE. We load a chunk of data, compute the max, then subtract max, exponentiate, sum, and then normalize.

Wait, but reductions across a row require communication between threads. Oh right, Triton has helper functions for reductions. For max, we can use tl.max along the columns. Similarly for sum.

Also, to handle varying sizes, the kernel should be written with proper masking to avoid out-of-bounds access when the dim_size isn't a multiple of BLOCK_SIZE.

Let's outline the kernel steps:

1. The kernel is launched with one program per row (batch_size blocks, each handling a row).
2. Each program (block) iterates over the columns in steps of BLOCK_SIZE.
3. For each chunk, load the data with masking.
4. Compute the row max using tl.max.
5. Subtract max from each element, exponentiate, and compute the sum.
6. Once all chunks are processed, normalize each element by the sum.

Now, let's code this. The Triton kernel will take the input tensor and output tensor. The meta-parameters include the number of columns (cols), and the BLOCK_SIZE.

Wait, the input tensor is 2D. So, each row is processed independently. The kernel's grid should be (batch_size, ), and each block handles a row.

The kernel function in Triton would look something like this:

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols

    # Load the row into SRAM, using mask to avoid out-of-bounds
    row = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    # Compute max
    row_max = tl.max(row, axis=0)
    # Subtract max for numerical stability
    row_minus_max = row - row_max
    # Exponentiate
    numerator = tl.exp(row_minus_max)
    # Compute sum
    row_sum = tl.sum(numerator, axis=0)
    # Divide to get softmax
    softmax_output = numerator / row_sum
    # Write back
    tl.store(output_ptr + offsets, softmax_output, mask=mask)

Wait, but the code above has a problem. If the row is processed in chunks, but the max and sum need to be computed across the entire row, not just a chunk. So the initial approach is incorrect. Because each chunk processes part of the row, but the max and sum need to be computed across all elements of the row.

Oh right! So in Triton, for reductions across the entire row, each program (block) handles the entire row. But if the row is larger than the BLOCK_SIZE, we need to process the row in multiple steps.

So the correct approach is to iterate through the row in chunks, compute the max and sum across all chunks, then go through the row again to compute the exponentials and divide by the sum.

Wait, that's more complex. So the kernel needs to perform two passes over the data. First, compute the max and sum, then compute the normalized values.

Alternatively, use two separate kernels. The first kernel computes the max and sum for each row, and the second applies the softmax. But that might be inefficient.

Alternatively, can we do it in a single kernel?

Yes, but it requires multiple steps. Let me think.

Each row is processed by a single block. The block processes the row in chunks. For each chunk, compute the local max and accumulate into a global max for the row. Similarly, compute the sum of exp(x_i - max) across all chunks.

So the steps are:

1. Initialize row_max to -infinity and row_sum to 0.
2. For each chunk in the row:
   a. Load the chunk elements.
   b. Compute the local max and update row_max.
3. After processing all chunks, compute the global max for the row.
4. Then, process each chunk again:
   a. Load the chunk elements.
   b. Subtract the global max, exponentiate.
   c. Compute the local sum and accumulate into row_sum.
5. After processing all chunks again, compute the global sum.
6. Finally, process each chunk a third time to divide by the global sum and store.

But this would require three passes over the data, which is not efficient.

Alternatively, can we fuse some steps? Let's see.

Actually, to compute the max and sum in a single pass, but that's not possible because the sum requires knowing the max first. So two passes are necessary.

Hmm. So the correct approach is to first compute the max of the entire row, then compute the sum of exp(x_i - max), and then compute the normalized values. But since in Triton, each block processes the entire row (possibly in chunks), this can be done in a single kernel.

So the sequence is:

1. For each row (handled by a block):
   a. Loop over chunks to compute the max.
   b. Loop over chunks again to compute the sum of exp(x - max).
   c. Loop over chunks again to compute exp(x - max) / sum and store.

But this requires three loops over the data. However, each loop processes the data in chunks, which may be manageable.

Alternatively, can we compute the max and sum in separate loops and reuse the data? Let's think.

The Triton kernel would need to load the data multiple times. Let's outline the code:

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Compute max
    max_val = -float('inf')
    for block_start in range(0, cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < cols
        row_chunk = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
        curr_max = tl.max(row_chunk, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    
    # Compute sum of exp(x - max)
    sum_val = 0.0
    for block_start in range(0, cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < cols
        row_chunk = tl.load(input_ptr + offsets, mask=mask, other=0)
        exp_x = tl.exp(row_chunk - max_val)
        sum_val += tl.sum(exp_x, axis=0)
    
    # Compute softmax and store
    for block_start in range(0, cols, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = block_start + tl.arange(0, BLOCK_SIZE) < cols
        row_chunk = tl.load(input_ptr + offsets, mask=mask, other=0)
        softmax_values = tl.exp(row_chunk - max_val) / sum_val
        tl.store(output_ptr + offsets, softmax_values, mask=mask)

But wait, in Triton, variables declared inside the kernel are per-thread unless using shared memory or atomic operations. So in the code above, max_val and sum_val are computed per-thread, which would be incorrect. Because each program (block) processes a row, but each thread in the block is handling part of the row. We need to perform reductions across the threads in the block.

Ah, right! This is a critical point. The code above is incorrect because each thread in the block is processing a different part of the row. Therefore, to compute the max and sum across all elements in the row, the block needs to perform a reduction across all threads.

So the correct approach is to have each block process a single row, and use block-wise reductions.

Let's adjust the code.

Each thread in the block will process certain elements of the row, then the block will compute the max and sum across all elements.

So the steps are:

1. Each block processes a row.
2. Each thread in the block loads a portion of the row (using BLOCK_SIZE as the number of elements per thread, maybe).
3. Compute the max across all threads in the block.
4. Compute the sum of exp(x - max) across all threads.
5. Then compute each element's softmax value and write back.

Wait, but this requires that all threads in the block collaborate to compute the max and sum. For this, we need to use Triton's reduction functions that can handle cross-thread reductions.

For example, when each thread has a local max, we can use tl.max to reduce across the block.

But Triton's reductions are along a specific axis. So perhaps using tl.reshape and then performing a reduction.

Alternatively, we can use tl.reduce to perform a reduction across the entire block.

Let me think. Let's structure it properly.

Each block is responsible for a single row. The block has a certain number of threads, say BLOCK_SIZE=1024. But if the row is larger than BLOCK_SIZE, then each thread can process multiple elements.

Wait, this is getting a bit complicated. Let's look for existing Triton softmax implementations.

Upon checking, Triton's tutorial has a softmax example. Let's recall that.

In the Triton tutorial, the softmax kernel is implemented as follows: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

They process each row by a single block. Each block first loads a row into shared memory, computes max, then subtracts max and computes the sum of exponentials, then divides each element by the sum. But this approach is suitable when the row fits into shared memory. If the row is large, this approach may not be feasible.

However, in our case, the input rows can be up to 4096 elements. So using a block size of 512 or 1024, each thread can process multiple elements. But for rows larger than the block size, the original tutorial code would not work.

Wait, the tutorial code uses a block size equal to the row size, which may not be feasible if the row is large. For example, if the row is 4096 elements, and the block size is 1024, then each thread would process 4 elements. But the code in the tutorial assumes that each thread handles one element.

So the tutorial code is designed for row sizes that are a multiple of the block size. To handle larger rows, we need to process them in chunks.

But in our case, the problem allows for varying dim_sizes (64 to 4096). So the kernel needs to handle rows of any size.

The correct approach is to process the row in chunks, compute the max and sum across all chunks using block reductions.

So here's the plan:

1. For each row (handled by a block), each thread processes multiple elements in chunks.
2. Compute the max across all elements in the row by iterating over chunks and accumulating the max via block reductions.
3. Similarly, compute the sum of exponentials (after subtracting max) by iterating over chunks again and accumulating via block reductions.
4. Finally, iterate over chunks again to compute the softmax values (exp(x - max) / sum) and store them.

So the Triton kernel will have three loops over the row's elements: first to compute the max, second to compute the sum, third to compute and store the softmax.

But each loop processes the row in chunks, and uses block reductions to accumulate the max and sum.

Now, implementing this in Triton.

Here's the code structure:

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Initialize max to -infinity
    row_max = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        current_mask = (block_start + col_offsets) < n_cols
        current_mask = current_mask & mask
        a = tl.load(input_ptr + offsets, mask=current_mask, other=-float('inf'))
        a = a.to(tl.float32)  # Work in float32 for precision
        row_max = tl.maximum(row_max, a)
    # Compute the max across the block
    max_val = tl.max(row_max, axis=0)

    # Compute sum of exp(x - max)
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        current_mask = (block_start + col_offsets) < n_cols
        current_mask = current_mask & mask
        a = tl.load(input_ptr + offsets, mask=current_mask, other=0)
        a = a.to(tl.float32)
        a_minus_max = a - max_val
        exp_a = tl.exp(a_minus_max)
        sum_exp += exp_a
    sum_val = tl.sum(sum_exp, axis=0)

    # Compute softmax and write back
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = row_start + block_start + col_offsets
        current_mask = (block_start + col_offsets) < n_cols
        current_mask = current_mask & mask
        a = tl.load(input_ptr + offsets, mask=current_mask, other=0)
        a = a.to(tl.float32)
        a_minus_max = a - max_val
        exp_a = tl.exp(a_minus_max)
        softmax_val = exp_a / sum_val
        tl.store(output_ptr + offsets, softmax_val.to(output_ptr.dtype.element_ty), mask=current_mask)

Wait, but in this code, each thread is accumulating a local sum of exp(x - max) over the chunks. Then, after the loop, we sum across the block. But this might not be correct. Let's see.

In the first loop (computing max), each thread processes a chunk and keeps track of the max in their local row_max array. Then, after the loop, the max across all elements is found by taking the max of row_max, which is the max across all threads in the block. That works.

In the sum_exp loop, each thread accumulates exp(x - max) into their local sum_exp array. Then, after the loop, sum_val is the sum across all elements by summing sum_exp across the block.

Yes, this should work. But each thread's sum_exp holds the sum of their chunks, and then the sum across threads gives the total sum for the row.

Now, the Triton kernel is written with three loops over the data. Each loop processes the row in chunks of BLOCK_SIZE elements. The first two loops compute the max and sum, and the third loop writes the softmax.

The BLOCK_SIZE is a tunable parameter. For best performance, it should be a power of two. Common values are 1024 or 512, depending on the GPU architecture.

The Triton function then calls this kernel with the appropriate grid and block dimensions. The grid is (batch_size, ), and each block handles one row. The block size (BLOCK_SIZE) is set to the minimum of the dim_size and the maximum possible block size (like 1024).

Wait, but in practice, the BLOCK_SIZE is a kernel parameter. So the function that calls the kernel should determine the appropriate BLOCK_SIZE. For example, setting BLOCK_SIZE to 1024, and if the row is smaller than that, the kernel's masking will handle it.

But for optimal performance, BLOCK_SIZE should be as large as possible but not exceed the maximum allowed (like 1024). So in the code, we can set BLOCK_SIZE to the next power of two of the dim_size, but dynamically setting it might be tricky. However, Triton can automatically tune the BLOCK_SIZE if we use a config with 'BLOCK_SIZE': triton.next_power_of_2(n_cols).

But in the code provided, the user wants the Triton function to be called as generated_triton_func, so the function should handle varying dim_sizes.

Putting it all together, the Python function that calls the Triton kernel would look like this:

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)  # Adjust based on GPU limits
    y = torch.empty_like(x)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 8
    # Define the kernel grid
    grid = (n_rows,)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y

But wait, the Triton kernel's BLOCK_SIZE is a constexpr, so it must be a compile-time constant. Therefore, this approach won't work as the BLOCK_SIZE varies per call. Oh, right. This is a problem. Because Triton kernels are compiled for specific BLOCK_SIZE values. So if we call the kernel with varying BLOCK_SIZE, each different size would require a separate compilation.

To handle this, the Triton kernel's BLOCK_SIZE must be a tunable parameter. However, in practice, for varying input sizes, the code may need multiple kernel launches with different BLOCK_SIZE values. Alternatively, set a fixed BLOCK_SIZE and pad the input, but that complicates things.

But in the problem statement, the input_generator uses varying batch and dim sizes. Therefore, the Triton kernel should be able to handle any dim_size, which suggests that we need a BLOCK_SIZE that is a multiple of the dim_size, but that's not feasible for varying sizes.

Alternatively, we can choose a fixed BLOCK_SIZE (like 1024) and handle any dim_size by processing in chunks. This is the approach taken in the code above. The code uses a fixed BLOCK_SIZE parameter (passed as a kernel argument), and uses loops to process the row in chunks. So the Triton kernel is written to handle any dim_size, as long as the BLOCK_SIZE is a power of two (for efficiency in reductions).

But how to choose BLOCK_SIZE optimally? Perhaps set it to 1024, as a default. However, for rows smaller than BLOCK_SIZE, the code will still work because of the masking.

Alternatively, use autotuning. But the user's code in the test section uses verify_correctness and benchmark_performance, which suggests that the Triton function should handle varying dim_sizes.

Therefore, the Triton kernel must use a fixed BLOCK_SIZE (like 1024) and process rows in chunks. The code above should work for any dim_size, as the for loop iterates over the row in steps of BLOCK_SIZE.

But in the code I outlined earlier, the kernel uses tl.arange(0, BLOCK_SIZE) for the column offsets. So even if the actual row has more elements than BLOCK_SIZE, the loop over block_start will process them in chunks.

Therefore, the kernel can handle any row size as long as it's called with a sufficiently large BLOCK_SIZE. However, choosing BLOCK_SIZE=1024 would handle up to 1024 elements per chunk. For rows larger than 1024, the loop will process multiple chunks.

Wait, no. If the row is 2048 elements and BLOCK_SIZE is 1024, the loop will run twice (0 and 1024). Each time, it loads 1024 elements. So the code is correct.

But then, when the row is 2048 and BLOCK_SIZE is 1024, each thread in the block will process two elements per loop iteration. Wait, no. The block size in Triton is the number of threads. If BLOCK_SIZE is set to 1024 in the kernel launch, then each thread processes one element per iteration. But if the row has more elements, the loop will handle multiple chunks.

Wait, no. The block size determines how many threads are in the block. Each thread in the block processes a different element in the chunk. For example, if the block has 1024 threads, each thread processes one element per chunk. The loop over block_start allows processing multiple chunks of 1024 elements.

But in the code above, the BLOCK_SIZE is a kernel parameter, which is a compile-time constant. So the kernel is compiled for a specific BLOCK_SIZE, like 1024. Then, when the kernel is launched, the loop over block_start in steps of BLOCK_SIZE processes the entire row.

So for a row of 2048 elements, with BLOCK_SIZE=1024, the loop would process two chunks: 0-1023 and 1024-2047. Each thread processes one element per chunk. But this requires that the block size (number of threads) is equal to BLOCK_SIZE. So each thread processes one element per chunk.

But Triton's program_id is the block index. For each row, the kernel is launched with a grid of (n_rows, ), and each block has BLOCK_SIZE threads. Each thread in the block handles one element per chunk.

Wait, no. Each block has a fixed number of threads, which is determined when the kernel is launched. For example, if the kernel is launched with block size (1024,), then each block has 1024 threads. The code inside the kernel uses tl.program_id(0) to get the row index, and each thread in the block processes elements in that row.

But in the code above, each thread handles one element per chunk. So for a row of 2048 elements and BLOCK_SIZE=1024, each thread would process two elements (two loop iterations). The first iteration processes elements 0-1023, the next processes 1024-2047.

But in Triton, each thread's ID is determined by tl.arange(0, BLOCK_SIZE). So for each loop iteration, the thread processes the element at block_start + thread_id.

Thus, the code should work for any row size, as the loop iterates through the row in steps of BLOCK_SIZE.

So the Triton kernel is written with a fixed BLOCK_SIZE (as a constexpr), and the code inside uses loops to process all elements of the row.

Now, the Python function that calls this kernel would need to set the BLOCK_SIZE appropriately. However, since BLOCK_SIZE is a compile-time constant, the kernel must be compiled for each possible BLOCK_SIZE used. To handle varying row sizes, this would require multiple kernel instances with different BLOCK_SIZE values, which is impractical.

Therefore, to handle varying row sizes efficiently, the Triton kernel should use a fixed BLOCK_SIZE, and the code within the kernel should process the row in chunks of that size, regardless of the actual row length. The code above handles this via the for loops and masking.

Therefore, the Triton kernel can be written with a fixed BLOCK_SIZE (like 1024), and the code will handle any row size, even if it's larger than BLOCK_SIZE. The for loops will process the row in chunks, and the masking ensures that out-of-bounds elements are ignored.

Thus, the Python function that calls the Triton kernel can set BLOCK_SIZE to 1024 (or another fixed value) and use that for all row sizes. The kernel will handle varying sizes via masking and loops.

Now, the PyTorch softmax reference function is straightforward. We can write:

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

Putting it all together, the code for softmax_triton.py would include the Triton kernel and the wrapper functions.

Finally, the test code uses verify_correctness and benchmark_performance, passing the generated_triton_func and generated_torch_func, which are triton_softmax and torch_softmax.

Now, I need to write the code with the detailed comments as per the user's instructions.

The Triton function's docstring should describe the steps, arguments, returns, and explanation.

For the Triton kernel:

Description: Computes the softmax of the input tensor along the last dimension using Triton.
Args:
    output_ptr (triton.Pointer): Output tensor pointer. Shape (batch_size, dim_size)
    input_ptr (triton.Pointer): Input tensor pointer. Shape (batch_size, dim_size)
    input_row_stride (int): Stride between rows in the input tensor.
    output_row_stride (int): Stride between rows in the output tensor.
    n_cols (int): Size of the last dimension (dim_size).
    BLOCK_SIZE (tl.constexpr): Number of elements processed per block iteration.

Returns: None. The output tensor is modified in-place.

Explanation:
    1. For each row, compute the maximum value across all elements.
    2. Subtract the maximum value from each element to avoid overflow.
    3. Exponentiate each element and compute the sum.
    4. Divide each exponentiated element by the sum to obtain softmax probabilities.

Similar structure for the Python wrapper function.

But the user's format requires the docstring for each Python and Triton function. The Triton kernel is inside the Python function as a JIT-decorated function. So the Python wrapper (triton_softmax) and the Triton kernel both need their own docstrings.

But the user's example shows a Python function's docstring. The Triton kernel is written inside the Python function and has its own docstring.

So in the code:

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Description: Computes the softmax of the input tensor along the last dimension using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim_size) with softmax applied.

    Explanation:
        1. Launches a Triton kernel to compute softmax for each row in parallel.
        2. Each kernel computes max, sum of exponentials, and normalization.
    """
    # ... code ...

@triton.jit
def softmax_kernel(...):
    """
    Description: [as before]
    Args: [as before]
    Returns: [as before]
    Explanation: [as before]
    """
    # ... code ...

Now, implementing all this into the code.

Finally, the test code uses the input_generator as specified, and calls verify_correctness and benchmark_performance with the generated functions and input generator.

So the code file softmax_triton.py should include:

- The Triton kernel function with JIT decorator.
- The triton_softmax Python function.
- The torch_softmax function.
- The code to call verify_correctness and benchmark_performance using the given parameters.

Wait, wait. The user's test code includes:

    verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);

But in Python, the syntax would be:

verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])

But the input_generator is defined as a function that takes batch_size, dim_size, device, and dtype.

So the code in softmax_triton.py must include the input_generator, but according to the user's instruction, the code should test using the provided code literally.

But the user's problem statement says to generate the code that uses the provided test code. So the generated code should have:

batch_sizes = [256, 512, 1024, 2048]
dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    return x
verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])

But in the generated code, the generated_triton_func is triton_softmax, and generated_torch_func is torch_softmax.

Putting it all together, the code in softmax_triton.py would look like:

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(...):
    # kernel code

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    # wrapper code

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    # test code here
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    from benchmark import verify_correctness, benchmark_performance
    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])

But the user's instruction states to call the functions in benchmark.py, which already exists. So the generated code should include the main block that calls verify_correctness and benchmark_performance with the given parameters.

Now, putting all this together with proper comments for each function as per the user's template.

The Triton kernel's docstring should follow the template with Description, Args, Returns, Explanation.

The softmax_kernel has parameters:

Args:
    output_ptr (triton.Pointer): Output tensor pointer. Shape (batch_size, dim_size)
    input_ptr (triton.Pointer): Input tensor pointer. Shape (batch_size, dim_size)
    input_row_stride (int): Stride between rows in the input tensor.
    output_row_stride (int): Stride between rows in the output tensor.
    n_cols (int): Number of columns (dim_size).
    BLOCK_SIZE (tl.constexpr): Number of elements processed per block.

The Explanation section lists the steps as before.

The triton_softmax function's docstring describes its purpose, arguments, and steps.

Now, the final step is to write the code with all these components and ensure proper formatting.
</think>

```python
# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Computes softmax probabilities along the last dimension of a 2D tensor.

    Args:
        output_ptr (triton.Pointer): Output tensor pointer. Shape (batch, dim)
        input_ptr (triton.Pointer): Input tensor pointer. Shape (batch, dim)
        input_row_stride (int): Stride between input tensor rows
        output_row_stride (int): Stride between output tensor rows
        n_cols (int): Number of columns in input tensor
        BLOCK_SIZE (tl.constexpr): Processing block size for kernel optimization

    Returns:
        None: Modifies output tensor in-place

    Explanation:
        1. Each program processes one row using program_id(0) for row index
        2. Iterate through columns in BLOCK_SIZE chunks to find row maximum
        3. Compute exponentials normalized by row maximum
        4. Calculate sum of exponentials across entire row
        5. Write normalized softmax values (exponential / sum) to output
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Find row max
    max_val = -float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        ptr = input_ptr + row_start + cols
        curr = tl.load(ptr, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(curr, axis=0))

    # Compute exp(x - max)
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        ptr = input_ptr + row_start + cols
        curr = tl.load(ptr, mask=mask, other=0)
        exp_val = tl.exp(curr - max_val)
        sum_exp += tl.sum(exp_val, axis=0)

    # Compute and store softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        ptr = input_ptr + row_start + cols
        curr = tl.load(ptr, mask=mask, other=0)
        softmax_val = tl.exp(curr - max_val) / sum_exp
        out_ptr = output_ptr + row_start + cols
        tl.store(out_ptr, softmax_val, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax using Triton optimized kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, dim)

    Returns:
        torch.Tensor: Output tensor with softmax applied along last dimension

    Explanation:
        1. Determines optimal block size based on input dimensions
        2. Allocates output tensor with same shape as input
        3. Launches triton kernel with grid_size = batch_size
        4. Each kernel instance processes one row of the input tensor
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    num_warps = 8 if BLOCK_SIZE >= 512 else 4
    
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies standard PyTorch softmax for verification.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, dim)

    Returns:
        torch.Tensor: Output tensor with softmax applied along last dimension
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    
    from benchmark import verify_correctness, benchmark_performance
    
    verify_correctness(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
```

<test>
```bash
python3 softmax_triton.py
```
</test>