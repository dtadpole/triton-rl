API: deepseek
Test Case: ./func_softmax/
Timestamp: 20250315_072214

<think>
Alright, let me try to figure out how to approach this problem. The user wants me to generate Triton code for a softmax function in a file called softmax_triton.py. They also provided a main block that tests correctness and performance using functions from benchmark.py. 

First, I need to recall how softmax works. Softmax applies the exponential function to each element and normalizes by the sum of exponentials across the dimension. The challenge in Triton is to do this efficiently, especially handling the reduction step across rows.

I remember that in Triton, we can process rows in parallel. Each program instance (block) would handle a row. The steps would be: load the row into SRAM, compute the maximum for numerical stability, compute exponentials and sum them, then normalize each element by the sum.

So, the Triton kernel would need to:
1. Iterate over each row.
2. Find the max value in the row.
3. Subtract the max from each element (for numerical stability).
4. Compute exp for each element and sum these values.
5. Divide each exp(element) by the sum to get the softmax.

I need to write the Triton function with proper indexing. Also, since the input is 2D (batch_size, dim_size), each row is a dimension of size dim_size. The kernel should handle different dim_sizes, so maybe using a BLOCK_SIZE that's a power of two and adjusting for when the dimension isn't a multiple of BLOCK_SIZE.

Wait, the user mentioned generated_triton_func and generated_torch_func. So I need to write both a Triton function and a PyTorch softmax for verification. The PyTorch function is straightforward using torch.nn.functional.softmax.

For the Triton kernel, the parameters would include the input tensor, and the output tensor. The kernel should be configured with grid and block sizes. Let's set the grid to have as many blocks as the number of rows (batch_size), and each block handles a row. The BLOCK_SIZE can be the next power of two up to 1024, but perhaps we can make it configurable based on the dim_size.

Wait, in the kernel, we need to handle varying dim_sizes. So maybe the BLOCK_SIZE is set to the nearest power of two that's at least as large as the dim_size. Or perhaps use a fixed BLOCK_SIZE and loop over the elements if the dimension is larger. Hmm, but Triton's efficiency comes from processing elements in blocks. Maybe using a BLOCK_SIZE of 1024, but then loop if the row is longer than that.

Alternatively, since the dim_sizes provided go up to 4096, maybe the BLOCK_SIZE can be 1024, and for each row, the kernel processes in chunks. For example, for a row of 4096, each block would process 1024 elements in four steps.

Wait, but for softmax, each row is processed independently. So each program instance (block) handles one row. The block's size (number of threads) should be a power of two. Let's set BLOCK_SIZE to 1024. Then, for rows smaller than that, we can mask the extra threads. For rows larger than BLOCK_SIZE, we can loop through the elements in chunks of BLOCK_SIZE.

Wait, no. Each row is a 1D tensor of size dim_size. So for each row, the kernel would load all elements, compute max, then compute sum of exps, then normalize. But if the dim_size is larger than the BLOCK_SIZE, we need to process the row in multiple steps. So the kernel would have a loop over the row in increments of BLOCK_SIZE. For example, for a row of 4096 elements, each program would process 1024 elements at a time (assuming BLOCK_SIZE=1024), accumulating the max and sum.

So the steps in the kernel would be:
1. For each row, iterate over elements in BLOCK_SIZE chunks.
2. Find the max value in the row by iterating through all elements.
3. Subtract the max from each element (for numerical stability) and compute exp for each, summing them up.
4. Then, normalize each element by the sum.

But how to handle the reduction (max and sum) across the row. For the max, each thread in the block loads a portion of the row, computes local max, then use a reduction across the threads to find the global max for the row. Similarly for the sum of exps.

Alternatively, using Triton's built-in reduction operations. Wait, Triton has functions like tl.reduce for reductions. For example, using tl.max to compute the maximum along the row.

So the kernel code outline might be:

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

Wait, but this assumes that the entire row fits into the BLOCK_SIZE. For rows larger than BLOCK_SIZE, this approach won't work. So the key is to handle rows of arbitrary length.

Therefore, we need to loop over the row's columns in chunks of BLOCK_SIZE. Let's adjust the kernel:

For each row, we process the columns in blocks of BLOCK_SIZE. For each block in the row, we load the elements, compute the max, then later compute the sum of exps, and finally compute each element's exp divided by the sum.

But how to compute the max and sum across the entire row when the row is processed in blocks. For that, we can first perform a first pass to compute the max of the entire row, then a second pass to compute the sum of exps (subtracting the max), and then a third pass to compute the softmax by dividing each exp by the sum.

But doing multiple passes over the data may be inefficient. Alternatively, can we compute the max and sum in a single pass by iterating over the blocks?

Hmm, here's the approach:

1. Each row is processed by a single program instance (program_id(0) is row index).
2. For the row, iterate over the columns in steps of BLOCK_SIZE.
3. For each block, load the elements, compute the local max, and keep track of the global max for the row.
4. Once the global max is known, iterate again over the blocks to compute the sum of exps (each element minus max, then exp, then sum).
5. Then, iterate over the blocks again to compute each element's exp divided by the sum and write to output.

But that requires three passes over the data, which could be expensive for large rows. However, given that the rows are processed in parallel, and each pass is sequential within the row, it may still be efficient.

Alternatively, find a way to compute the max and sum in a single pass. However, that's not possible because the sum of exps depends on knowing the max first. So, two passes are necessary: one for the max, and another for the sum and writing the output.

Wait, perhaps we can compute the max in the first pass, then compute the sum in a second pass, and then compute the normalized values in a third pass.

But that would be three passes. Alternatively, can we combine the sum and normalization into one pass?

Yes. Here's the plan:

1. First pass: compute the row max.
2. Second pass: compute the sum of exps (after subtracting the max) and store the exp values in a temporary array.
3. Third pass: divide each exp value in the temporary array by the sum and write to the output.

But this approach requires storing the exp values, which may be memory-intensive for large rows.

Alternatively, during the second pass, compute the sum, and during the same pass, compute the exp values and store them, then have a third pass to divide by the sum. But that's similar.

Alternatively, can we compute the sum and the exp values in one pass, then have a second pass for the division?

But in Triton, how to manage temporary storage? Maybe using shared memory or registers. But for large rows, this may not be feasible.

Alternatively, perhaps the approach is to split the kernel into two parts: compute the max, compute the sum, then compute the output. But how to manage intermediate values between different kernel launches.

Alternatively, use two kernels: one to compute the max and sum per row, and another to compute the softmax using those values. But that introduces synchronization and data transfer overhead.

Hmm. Given the constraints, perhaps the best approach for Triton is to handle rows that fit into a single block. For rows longer than BLOCK_SIZE, split into multiple blocks and use atomic operations to track the max and sum. But Triton may not support atomics for all operations, and this could complicate things.

Wait, but the user's test cases include dim_sizes up to 4096, which is 4 times the typical block size of 1024. So the kernel must handle rows longer than the block size.

So, here's a possible approach for the Triton kernel:

For each row:

1. Split the row into chunks of BLOCK_SIZE.
2. In the first loop, compute the max for each chunk and keep track of the overall max for the row.
3. Then, in a second loop over the chunks, compute the exp of each element minus the max, accumulate the sum of these exps.
4. Finally, in a third loop, compute each element's exp divided by the sum and store the result.

But how to manage the max and sum across the chunks. For example, when processing each chunk, the max for the row is the overall max, which may come from any chunk.

So, the first loop would iterate over the chunks, read each element in the chunk, compute the max for the chunk, and then compare with the overall max. Then, after processing all chunks, the overall max is known.

But in Triton, how to accumulate the max across different chunks. Since each chunk is processed in the same program, perhaps using a loop over the chunks and keeping track of the max in a register.

Wait, here's a possible implementation outline:

@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    # Compute max
    row_max = -float('inf')
    for col_block in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_idx * n_cols + col_offsets
        vals = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    # Compute sum
    row_sum = 0.0
    for col_block in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_idx * n_cols + col_offsets
        vals = tl.load(input_ptrs, mask=mask, other=0)
        exps = tl.exp(vals - row_max)
        row_sum += tl.sum(exps, axis=0)
    # Compute softmax
    for col_block in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_block + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_idx * n_cols + col_offsets
        vals = tl.load(input_ptrs, mask=mask, other=0)
        exps = tl.exp(vals - row_max)
        softmax = exps / row_sum
        output_ptrs = output_ptr + row_idx * n_cols + col_offsets
        tl.store(output_ptrs, softmax, mask=mask)

But wait, in Triton, the program is executed by a group of threads (a block), but in this case, each program is handling a single row. So the grid is (batch_size, 1), and the block size is 1. Wait no, the grid is determined by the number of rows (batch_size), and each program handles a single row. So the triton.jit function is called with grid=(batch_size,), and each program instance (row) processes its own row.

In this approach, each program loops through the columns of the row in chunks of BLOCK_SIZE. For each chunk, it loads the values, computes the max, then sum, then the softmax. The loops allow handling rows longer than BLOCK_SIZE.

This seems plausible. The BLOCK_SIZE is a tunable parameter. Let's set it to 1024, but the user can adjust it.

Now, for the Triton function, the code would wrap this kernel. The function would allocate the output tensor, and configure the kernel launch with the correct grid and block dimensions.

The generated_triton_func would be a Python function that calls this kernel. Similarly, the generated_torch_func is the PyTorch softmax.

So, putting it all together, the softmax_triton.py file would contain:

- The Triton kernel function with the loops as above.
- A Python function that calls the kernel, handling the tensor dimensions and kernel configuration.
- The PyTorch softmax function for verification.

But the user's code in the main block refers to generated_triton_func and generated_torch_func. So in the code, perhaps the Triton function is called 'softmax_triton' and the PyTorch function is 'softmax_torch'.

So, the code structure would be:

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(...):
    ...

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    # Allocate output
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    # Alternatively, set to 1024 and handle with loops
    # Wait, in the kernel, the BLOCK_SIZE is a constexpr, which must be a power of two.
    # So perhaps fixed BLOCK_SIZE of 1024, and handle any n_cols via loops.
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, n_cols, BLOCK_SIZE=1024)
    return y

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)

Then, in the main block, the code uses verify_correctness_func and benchmark_performance_func with these functions.

Wait, but the generated_triton_func and generated_torch_func in the main code are supposed to be the functions to test. So in the code, the functions are named softmax_triton and softmax_torch, and are passed to the benchmark functions.

So the main code will have:

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048];
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])

But in the code provided by the user, the functions are called generated_triton_func and generated_torch_func. So in the code, those should be replaced with the actual functions, softmax_triton and softmax_torch.

Now, for the Triton kernel, the code I outlined earlier may have a problem with the grid and block configuration. Each instance of the kernel processes a single row, so the grid is (n_rows,), and each block is a single thread? Or perhaps using 1D blocks. Wait, Triton uses block grids similar to CUDA, but the exact configuration is handled by the kernel's program_id.

In the kernel, each program_id(0) corresponds to a row. So the grid is (n_rows,) and the block size is (1,). But in Triton, the block size (number of threads per block) can be set to 1. However, using a block size larger than 1 may allow for more parallelism within a row, but the current approach processes each row sequentially in a single thread. That may not be efficient for large rows.

Alternatively, perhaps each row is processed by a block of threads. For example, if each block has 256 threads, then for a row of 4096 elements, the block would process 16 elements per thread (4096 / 256). But integrating that with the Triton kernel requires more complex code.

But given the time constraints and the user's request for a Triton implementation, perhaps proceed with the initial approach where each row is processed by a single thread, looping over the elements in chunks. However, this may not be the most efficient, especially for large rows, but it's a starting point.

Alternatively, here's a better approach inspired by the Triton documentation's softmax example. Let's look up how the Triton tutorial implements softmax.

In the Triton tutorial, they have a fused softmax example here: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

The tutorial's approach processes each row in a block, and each block handles the entire row. The key steps are:

1. Load the row into shared memory.
2. Compute the max by iterating over the elements and reducing.
3. Compute the sum of exps using the max.
4. Normalize each element and store.

This approach works when the row fits into shared memory, which is limited. For very large rows (e.g., 4096 elements), this may not be feasible. However, in their example, they use a BLOCK_SIZE of 1024, and process the row in chunks if necessary.

Wait, in the example, they have a BLOCK_SIZE of 1024, and the code handles rows of any size by looping over the columns in steps of BLOCK_SIZE. Let's adapt that code.

The tutorial's code for softmax_kernel:

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., like __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

But this requires that the row can be processed in a single block. The comment says that BLOCK_SIZE is the next power of two greater than n_cols. However, in the code, when launching the kernel, they compute the max_block_size as the next power of two up to n_cols, then set BLOCK_SIZE to max_block_size.

But if n_cols is 4096, then BLOCK_SIZE would be 4096, which may exceed the maximum allowed block size (1024 for Triton, depending on the GPU). So this approach may not work for large rows.

Alternatively, the tutorial code is designed for rows that can fit into a single block. For larger rows, this code would not work. Hence, the code in the tutorial is limited to rows up to 1024 elements (assuming BLOCK_SIZE is capped at 1024).

But the user's test cases have dim_sizes up to 4096, which exceeds typical block sizes. Therefore, the tutorial's code would not work for those cases. So, we need to adjust the kernel to handle rows larger than BLOCK_SIZE.

To handle larger rows, we need to process each row in multiple blocks. However, this complicates the reduction operations (max and sum) across the entire row. One approach is to use two passes: first to compute the max and sum, then to compute the softmax. But that would require multiple kernel launches.

Alternatively, use a single kernel that processes the row in multiple stages. For example, each program instance handles a row and processes it in chunks, accumulating the max and sum across the chunks.

This is what I was thinking earlier. Adapting the tutorial's code to handle rows larger than BLOCK_SIZE by using loops.

So, modifying the kernel to loop over the row's elements in chunks of BLOCK_SIZE.

Here's an updated kernel:

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    
    # Compute max
    max_val = -tl.infinity(tl.float32)
    for col_block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_start + col_offsets
        current_val = tl.load(input_ptrs, mask=mask, other=-tl.infinity(tl.float32))
        current_max = tl.max(current_val, axis=0)
        max_val = tl.maximum(max_val, current_max)
    
    # Compute sum of exp
    sum_exp = 0.0
    for col_block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_start + col_offsets
        current_val = tl.load(input_ptrs, mask=mask, other=0.0)
        current_val_minus_max = current_val - max_val
        current_exp = tl.exp(current_val_minus_max)
        sum_exp += tl.sum(current_exp, axis=0)
    
    # Compute softmax
    for col_block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = col_block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        input_ptrs = input_ptr + row_start + col_offsets
        current_val = tl.load(input_ptrs, mask=mask, other=0.0)
        current_val_minus_max = current_val - max_val
        current_exp = tl.exp(current_val_minus_max)
        softmax_val = current_exp / sum_exp
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        tl.store(output_ptrs, softmax_val, mask=mask)

This kernel loops over the row in chunks of BLOCK_SIZE. For each chunk, it computes the max, then the sum of exps, and finally the softmax values. This approach allows rows of any size, as long as the kernel can process them in multiple loops.

Now, in the Python function that calls this kernel, we need to set the grid and block size. The grid would be (n_rows,), and each block processes a row. The BLOCK_SIZE is set to a fixed value, say 1024, as a trade-off between parallelism and resource usage.

The function would be:

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024  # can be tuned
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

But wait, the input and output row strides are passed as parameters. This accounts for non-contiguous tensors. For a contiguous tensor, x.stride(0) would be n_cols, but using the actual stride allows handling of non-contiguous cases.

Now, the PyTorch function is straightforward:

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)

Putting it all together, the code in softmax_triton.py would include the kernel and these two functions. The main block is as provided by the user.

Now, regarding the documentation comments as per the user's request:

For the Triton function softmax_triton, the comment would describe the function's purpose, arguments, returns, and steps.

But according to the user's instruction, each Python and Triton function should have a detailed comment section. The Triton kernel is written in Triton's DSL, which is Python-decorated, but its arguments are in Triton's syntax. So, the comment for the Triton kernel would need to follow the same template.

For example:

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
) -> None:
    """
    Computes the softmax for each row of the input tensor.

    Args:
        output_ptr (triton.Pointer): Pointer to the output tensor. Shape (n_rows, n_cols)
        input_ptr (triton.Pointer): Pointer to the input tensor. Shape (n_rows, n_cols)
        input_row_stride (int): Stride between consecutive rows in the input tensor.
        output_row_stride (int): Stride between consecutive rows in the output tensor.
        n_cols (int): Number of columns in each row.
        BLOCK_SIZE (tl.constexpr): Block size for processing elements in chunks.

    Returns:
        None: The output is stored in the output_ptr tensor.

    Explanation:
        1. For each row, iterate over columns in chunks of BLOCK_SIZE.
        2. Compute the maximum value in the row for numerical stability.
        3. Subtract the maximum value from each element and compute the sum of exponentials.
        4. Divide each exponentiated element by the sum to get the softmax values.
        5. Store the results in the output tensor.
    """
    # code here

Similarly, the Python functions would have their own documentation.

The softmax_triton function's comment:

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax to each row of the input tensor using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor of the same shape as input, with softmax applied row-wise.

    Explanation:
        1. Determines the grid size based on the number of rows.
        2. Launches the Triton kernel with appropriate configuration.
        3. The kernel processes each row to compute the softmax.
    """
    # code here

The softmax_torch function's comment:

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax to each row of the input tensor using PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor of the same shape as input, with softmax applied row-wise.

    Explanation:
        1. Uses PyTorch's F.softmax function with dim=-1.
    """
    return torch.nn.functional.softmax(x, dim=-1)

Now, putting all the pieces together, the code for softmax_triton.py would look like this with proper comments.
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
    Computes row-wise softmax using three-step Triton kernel.

    Args:
        output_ptr (triton.Pointer): Output tensor pointer. Shape (n_rows, n_cols)
        input_ptr (triton.Pointer): Input tensor pointer. Shape (n_rows, n_cols)
        input_row_stride (int): Stride between input rows (elements)
        output_row_stride (int): Stride between output rows (elements)
        n_cols (int): Number of columns per row (dim_size)
        BLOCK_SIZE (tl.constexpr): Processing block size (triton constant)

    Returns:
        None: Output is written directly to memory

    Explanation:
        1. Iterate through row in BLOCK_SIZE chunks to find row maximum
        2. Second pass calculates sum of exponentials (max stabilized)
        3. Final pass computes and stores normalized softmax values
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Step 1: Find row maximum
    max_val = -tl.infinity(tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        ptrs = input_ptr + row_start + cols
        vals = tl.load(ptrs, mask=mask, other=-tl.infinity(tl.float32))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Step 2: Compute sum of exponentials
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        ptrs = input_ptr + row_start + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Step 3: Compute and store softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        ptrs = input_ptr + row_start + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        softmax_vals = tl.exp(vals - max_val) / sum_exp
        
        out_ptrs = output_ptr + row_idx * output_row_stride + cols
        tl.store(out_ptrs, softmax_vals, mask=mask)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of row-wise softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax-normalized tensor of same shape as input

    Explanation:
        1. Allocates output tensor matching input shape
        2. Configures kernel launch parameters (grid, block size)
        3. Invokes Triton kernel with input/output tensor metadata
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of row-wise softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
    
    Returns:
        torch.Tensor: Softmax-normalized tensor of same shape as input
    
    Explanation:
        1. Directly applies PyTorch's softmax along last dimension
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
```

<test>
To validate correctness and benchmark performance, run:

```bash
python softmax_triton.py
```

This will:
1. Verify numerical equivalence between Triton and PyTorch implementations across specified batch/dim sizes
2. Measure execution time and throughput for both implementations
3. Print verification results and performance comparison metrics
</test>