API: deepseek
Test Case: ./func_softmax/
Timestamp: 20250303_053915

<think>
Okay, I need to generate a Triton implementation of the softmax function. Let me start by recalling how softmax works. The softmax function takes a vector and converts it into probabilities by exponentiating each element and dividing by the sum of exponentials. The key steps are: compute the maximum for numerical stability, subtract this max, exponentiate, sum the exponentials, and then divide each element by this sum.

First, I'll need to write a Triton kernel. Triton uses blocks and grids similar to CUDA, but with some abstractions. Each program should handle a block of elements. For softmax, it's efficient to process rows of the input matrix. So, each block could handle a row. The input is a 2D tensor of shape (batch_size, dim_size).

The kernel needs to compute the max of each row, then compute the sum of exponentials after subtracting the max, and finally divide each element by this sum. Since Triton requires explicit memory management, I'll need to load the data into SRAM, perform the computations, and then write back the results.

For the max computation: each program loads a row, iterates through the elements to find the max. But wait, maybe using a reduction would be better. Triton has reduction operations, so perhaps using tl.max along the row dimension.

Once the max is found, subtract it from each element and exponentiate. Then compute the sum of these exponentials. Again, a reduction over the row to get the sum.

Then, each element is divided by the sum. All these steps need to be done in the kernel.

I should also write a PyTorch softmax function for correctness checking and benchmarking. The PyTorch version is straightforward using torch.nn.functional.softmax.

Now, looking at the provided main code, the input generator creates a tensor of shape (batch_size, dim_size). The Triton function needs to handle this shape. The Triton kernel will be launched with a grid of batch_size blocks, each handling a row of dim_size elements.

Wait, how to handle the grid and blocks? For each row in the batch, one Triton program. So the grid size is (batch_size,), and each block processes a row. The block size (number of threads per block) should be a power of two. Since the rows can be up to 4096 elements long, using 1024 threads per block might be efficient. But I need to split the row into multiple steps if the dim_size exceeds the block size.

Alternatively, using a 1D block where each thread handles multiple elements. For example, if the row size is 4096 and the block has 512 threads, each thread would process 8 elements (4096/512 = 8). So, using a loop over the elements in the row, with a stride equal to the block size.

So the kernel function in Triton would take pointers to the input, output, and the row stride (dim_size). The function would compute the max, subtract, exponentiate, sum, then divide.

Let me outline the steps:

1. Load the row data into a block.
2. Compute the max value in the row using a reduction.
3. Subtract the max from each element to avoid overflow.
4. Exponentiate each element.
5. Compute the sum of the exponentiated values.
6. Divide each exponentiated element by the sum to get the softmax probabilities.

In Triton code, each of these steps requires using tl.load and tl.store with appropriate masks. Also, using tl.max and tl.sum for reductions.

Now, writing the Triton kernel. The kernel function will be decorated with @triton.jit. The function parameters include input and output pointers, along with meta-parameters like N_COLS (the number of columns, i.e., dim_size), BLOCK_SIZE (the block size for processing).

The kernel should use program_ids along the rows (batch_size). Each program handles one row. The block is of size BLOCK_SIZE, which is a Triton parameter. For each element in the row, the thread (pid) will have to iterate through the row using offsets.

Wait, let's think about the code structure:

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    input_ptrs = input_ptr + row_start + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Compute max
    row_minus_max = row - tl.max(row, axis=0)
    row_exp = tl.exp(row_minus_max)
    exp_sum = tl.sum(row_exp, axis=0)
    softmax_output = row_exp / exp_sum

    # Write back
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

Wait, but this code is for when the entire row fits into a single block. If the row is larger than BLOCK_SIZE, this approach won't work. So, the kernel needs to handle rows that are longer than the block size by iterating over chunks.

Ah, right, so for rows longer than BLOCK_SIZE, we need to process the row in chunks. So the code would load each chunk, compute the max of the chunk, keep track of the overall max, then repeat for all chunks. Then, subtract the max from each element, exponentiate, compute the sum, and then divide.

So the correct approach is:

1. Each block processes a row. If the row is longer than BLOCK_SIZE, the block loops through the row in steps of BLOCK_SIZE.
2. For each chunk in the row, compute the local max, then update the global max.
3. Once the global max is found for the entire row, each element is subtracted by the max and exponentiated.
4. Then, compute the sum of the exponents by iterating through the row again.
5. Finally, divide each exponentiated element by the sum and store the result.

This requires two passes over the data: first to compute the max, and second to compute the sum of exponents. Alternatively, can we compute the max and sum in a single pass? No, because the sum depends on the max. So two passes are needed.

Wait, no. The first pass computes the max. Then, after subtracting the max, the second pass computes the sum of exponents. But to compute the sum, we need to process all elements again. So it's two passes over the data.

But how to handle this in Triton? The kernel would need to load the data twice. Let me structure the code accordingly.

So the kernel would:

First pass (compute max):

- Initialize max_val to -infinity.
- Loop over the row in chunks of BLOCK_SIZE.
- For each chunk, load the elements, compute the max of the chunk, and compare with the current max_val, updating if necessary.
- After processing all chunks, the max_val is the maximum of the entire row.

Second pass (compute sum of exp(row - max_val)):

- Initialize sum_val to 0.
- Loop over the row again, load each chunk, compute exp(row - max_val), add to sum_val.

Third pass (compute softmax):

- Loop over the row, compute (exp(row - max_val) / sum_val) and store in the output.

But this requires three passes over the data, which can be inefficient. Alternatively, can we combine the sum computation and the softmax application? Let's think.

After computing the max, we can compute the sum of exp(row - max) by looping through the elements once. Then, in the same loop, compute exp(row - max) and store them in a temporary array, then divide each by the sum. But Triton doesn't allow dynamic memory allocation, so we can't store the exponents. Hmm.

Alternatively, perform two passes: first, compute max, then compute sum and exponents. Wait, but to compute the sum, we need to process all elements and compute their exponent after subtracting the max. Then, in a third pass, divide each exponent by the sum.

This would require three passes. But if the data is not cached, that's a lot of memory accesses. However, in practice, for large rows, this might still be manageable.

Another approach is to use shared memory to store the intermediate exponents. But Triton abstracts away shared memory, so perhaps it's not necessary.

Alternatively, find a way to compute the sum in the same pass as computing the exponents. But since the sum depends on all elements, it's not possible unless using a reduction.

Wait, perhaps in the second pass, after subtracting the max, compute the exp and accumulate the sum in one loop. Then, in a third loop, read the elements again and divide by the sum. But again, that's three passes.

Alternatively, we can compute the sum in one loop, store the exponents, then in another loop divide. But in Triton, how to manage the storage? Since Triton kernels don't have shared memory across the entire kernel execution (each block's shared memory is per-block?), perhaps this isn't feasible.

Alternatively, for each element in the row, during the sum pass, compute exp(x - max), add to sum (using a reduction), and store exp(x - max) in a temporary buffer. Then, divide each element in the buffer by the sum. But storing the temporary buffer requires memory. However, in Triton, each program's local variables are per-thread, so we can't have a buffer that's shared across the entire row processing.

Hmm, this is getting complicated. Perhaps the Triton kernel will have to process each element twice: once to compute the max, once to compute the sum and apply the division.

Alternatively, given that each row is processed by one block, and the block can load the data multiple times, maybe the code can be structured as follows:

1. Compute the max of the entire row by loading chunks and reducing.
2. Compute the sum of exp(x - max) by loading each chunk again and reducing.
3. Compute the softmax by loading each chunk again, compute exp(x - max) / sum, and store.

This requires three loops over the data. But for large rows, this might be inefficient. However, given that each row is handled by one block, and Triton handles the parallelism, this might be manageable.

Let's proceed with this approach.

So the Triton kernel code would look something like this:

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Step 1: Compute max
    max_val = tl.zeros((BLOCK_SIZE,), tl.float32) - float('inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        curr_max = tl.max(row, axis=0)
        max_val = tl.maximum(max_val, curr_max)
    max_val = tl.max(max_val, axis=0)

    # Step 2: Compute sum of exp(row - max)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        row = tl.load(input_ptrs, mask=mask, other=0.0)
        row_minus_max = row - max_val
        row_exp = tl.exp(row_minus_max)
        sum_exp += tl.sum(row_exp, axis=0)
    
    # Step 3: Compute softmax and store
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        output_ptrs = output_ptr + row_start + cols
        row = tl.load(input_ptrs, mask=mask, other=0.0)
        row_minus_max = row - max_val
        row_exp = tl.exp(row_minus_max)
        softmax = row_exp / sum_exp
        tl.store(output_ptrs, softmax, mask=mask)

Wait, but in this code, each step loops through the row in blocks. So for each step (max, sum, store), the code is loading the data again. That's three passes. For large rows, this could be memory intensive.

But perhaps this is the correct approach. Triton's compiler might optimize these loops. Let's proceed.

Now, the Triton function needs to be called with the right parameters. The kernel launch parameters should have grid = (batch_size,) and block = (BLOCK_SIZE,), but wait, the block size is 1D here. But in the kernel, each program is handling a row, so the grid is (batch_size, 1, 1), and each block is (1, 1, 1)? Wait no. The kernel's program_id(0) corresponds to the row index. The block is a 1D block of size BLOCK_SIZE. Wait, no: in Triton, the grid and block are 1D, 2D, or 3D, similar to CUDA. Each program in the grid is a single instance of the kernel. So for processing each row, each row is handled by one program. The block size in this context is the number of threads (or Triton's equivalent) per program. Wait, perhaps I'm mixing CUDA concepts with Triton. In Triton, the grid is the number of program instances, and each instance processes a block of data.

Wait, the Triton documentation says that each element in the grid is a program instance. Each program can process a block of data using loops and vector operations. So for a row-wise operation, each program handles one row. The block size here refers to the number of elements processed in each chunk. For example, if the row has 4096 elements and BLOCK_SIZE is 1024, each program will loop 4 times (4096 / 1024) to process the entire row.

So the grid is (batch_size,), and the block size (BLOCK_SIZE) is the number of elements processed per loop iteration. The program is designed to process each row in chunks of BLOCK_SIZE elements.

Now, the kernel code loops over the row in steps of BLOCK_SIZE, processing each chunk. Each time, it loads a chunk of size BLOCK_SIZE, applies operations, and accumulates the max or sum.

The max is computed by first initializing max_val to -infinity, then loading each chunk and taking the max between the current max and the chunk's max. Then, after processing all chunks, the max_val is the maximum of the entire row.

Similarly, for the sum, after subtracting the max, each chunk is exponentiated and summed, and the sum is accumulated.

Finally, in the third loop, each chunk is loaded again, exponentiated, divided by the sum, and stored.

This code should handle rows of any size, as long as they are multiples of BLOCK_SIZE, or with the mask handling.

Now, the Triton kernel is written. But how to call it from Python? The Python function will need to allocate the output tensor, determine the BLOCK_SIZE, and launch the kernel.

The BLOCK_SIZE should be a power of two for efficient reductions. Let's choose the next power of two larger than the row size, but up to 1024, which is a common block size. Wait, but in the code, the BLOCK_SIZE is the size of each chunk processed in a loop. So for rows larger than BLOCK_SIZE, the code will loop multiple times to cover the entire row. For example, for a row of size 4096 and BLOCK_SIZE 1024, each loop iteration processes 1024 elements, and there are 4 iterations.

But how to choose BLOCK_SIZE? It's a tunable parameter. But the user can set it to the maximum possible, such as 1024. However, the code should use the minimum between the row size and the BLOCK_SIZE. Wait, the code already handles that by looping and using the mask.

So the Python function will call the kernel with a grid of (batch_size,), and each program will process a row. The BLOCK_SIZE is set to the next power of two up to 1024 or the row size, whichever is smaller. But to avoid complexity, perhaps set BLOCK_SIZE to 1024, and let the code loop as necessary.

The Python code would look like this:

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel( ... ):
    ... # as above

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    batch_size, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    y = torch.empty_like(x)
    softmax_kernel[(batch_size,)](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

Then, the generated_triton_func is triton_softmax, and the generated_torch_func is torch.nn.functional.softmax with dim=-1.

But wait, the input to the Triton kernel is x, and the output is y. The kernel is launched with grid size (batch_size,), and each program handles one row. The BLOCK_SIZE is determined based on the row length.

Now, the Triton kernel's parameters include input_row_stride and output_row_stride, which are the strides for the input and output tensors. For a contiguous tensor, x.stride(0) is the number of elements between rows, which would be n_cols. However, if the tensor is not contiguous, the stride would account for that. The kernel uses row_start = row_idx * input_row_stride to get the starting index for each row.

This should handle non-contiguous tensors as well, as long as the stride is correctly passed.

Now, the test code:

In the main block, they generate inputs using input_generator, which returns x of shape (batch_size, dim_size). They call verify_correctness_func and benchmark_performance_func with generated_triton_func and generated_torch_func.

So the generated_torch_func is a function that applies torch softmax. Let's write that as:

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)

Therefore, in the main code:

generated_triton_func = triton_softmax
generated_torch_func = torch_softmax

But need to make sure that the Triton function and Torch function are compatible in terms of input and output.

Now, putting it all together.

The code in softmax_triton.py would be:

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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Description:
        Computes the softmax for each row in the input matrix using Triton.

    Args:
        output_ptr (triton.PointerType): Pointer to the output tensor. Shape (batch_size, dim_size)
        input_ptr (triton.PointerType): Pointer to the input tensor. Shape (batch_size, dim_size)
        input_row_stride (int): Stride in elements between consecutive rows in the input.
        output_row_stride (int): Stride in elements between consecutive rows in the output.
        n_cols (int): Number of columns (dim_size) in each row.
        BLOCK_SIZE (tl.constexpr): Block size for processing elements in a row.

    Returns:
        None: The output is stored in the output_ptr.

    Explanation:
        1. Each program processes a single row of the input matrix.
        2. Compute the maximum value in the row for numerical stability.
        3. Subtract the maximum value from each element and exponentiate.
        4. Compute the sum of exponentiated values.
        5. Divide each exponentiated element by the sum to obtain softmax probabilities.
        6. Store the results in the output tensor.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Step 1: Compute max value in the row
    max_val = -float('inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        curr = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        curr_max = tl.max(curr, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # Step 2: Compute sum of exp(row - max)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        curr = tl.load(input_ptrs, mask=mask, other=0.0)
        curr = curr - max_val
        exp = tl.exp(curr)
        sum_exp += tl.sum(exp, axis=0)

    # Step 3: Compute softmax and store result
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        output_ptrs = output_ptr + row_idx * output_row_stride + cols
        curr = tl.load(input_ptrs, mask=mask)
        curr = curr - max_val
        exp = tl.exp(curr)
        softmax = exp / sum_exp
        tl.store(output_ptrs, softmax, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Applies softmax to each row of the input tensor using a Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor of the same shape as input, containing softmax probabilities.

    Explanation:
        1. Determine the optimal block size for the Triton kernel.
        2. Allocate output tensor.
        3. Launch the softmax_kernel with appropriate grid and block dimensions.
        4. Return the output tensor.
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Description:
        Applies softmax to each row of the input tensor using PyTorch's softmax function.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).

    Returns:
        torch.Tensor: Output tensor of the same shape as input, containing softmax probabilities.

    Explanation:
        1. Applies PyTorch's F.softmax along the last dimension.
    """
    return torch.nn.functional.softmax(x, dim=-1)

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

Wait, the code has a few things to check. For example, the BLOCK_SIZE calculation in the triton_softmax function uses next_power_of_2, but then is clamped between 128 and 4096. That's to ensure it's within a reasonable range. However, this might need tuning for optimal performance. But it's a starting point.

In the Triton kernel, the first step computes the max. For each chunk, it loads the elements and computes the max of that chunk, then updates the overall max. The initial max_val is set to -inf. Then, after processing all chunks, the max_val is the maximum of all elements.

Then, the sum of exp(row - max) is computed by looping through each chunk, subtracting the max, exponentiating, and summing. This sum is accumulated.

Finally, each chunk is loaded again, exponentiated, divided by the sum, and stored.

This code should handle all row sizes, as the loop steps are based on BLOCK_SIZE and the mask ensures only valid elements are processed.

Testing this code with the provided main block should verify correctness and benchmark performance between the Triton and PyTorch implementations.
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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes softmax probabilities for each row in a 2D tensor.

    Args:
        output_ptr (triton.PointerType): Output tensor pointer. Shape (batch_size, dim_size)
        input_ptr (triton.PointerType): Input tensor pointer. Shape (batch_size, dim_size)
        input_row_stride (int): Input tensor row stride in elements
        output_row_stride (int): Output tensor row stride in elements
        n_cols (int): Number of columns/dimension size
        BLOCK_SIZE (tl.constexpr): Processing block size (triton constant)

    Returns:
        None: Results stored directly in output_ptr

    Explanation:
        1. Each program processes one row using program_id(0)
        2. Three-phase computation:
           a. Find row maximum for numerical stability
           b. Compute sum of exponentials after subtracting max
           c. Calculate final softmax values by dividing exp(x) by sum
        3. Uses block-wise processing for memory efficiency
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Phase 1: Compute row max
    max_val = -float('inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Phase 2: Compute exponential sum
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Phase 3: Compute final softmax
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask)
        exp_vals = tl.exp(vals - max_val)
        softmax_out = exp_vals / sum_exp
        tl.store(output_ptr + row_idx * output_row_stride + cols, softmax_out, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated softmax implementation for 2D tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax probabilities with same shape as input

    Explanation:
        1. Auto-tunes block size based on column dimension
        2. Launches parallel kernels for each row
        3. Uses Torch tensor memory layout directly
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(min(BLOCK_SIZE, 1024), 128)  # Tuned performance constraints
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Reference PyTorch softmax implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax probabilities with same shape as input
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

    from benchmark import verify_correctness_func, benchmark_performance_func

    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
```

<test>
```bash
python softmax_triton.py
```
</test>