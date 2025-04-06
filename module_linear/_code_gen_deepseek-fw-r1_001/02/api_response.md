Timestamp: 20250406_043845

<think>
Okay, I need to generate a Triton module called TritonLinear that implements both forward and backward passes for a PyTorch nn.linear equivalent. The user mentioned using autotune for block sizes like BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O in both forward and backward kernels, along with super blocks and autotuning GROUP_SIZE_* parameters. Also, they provided a test script that uses benchmark.py to verify correctness and performance.

First, I should start by defining the TritonLinear class. It needs to inherit from torch.nn.Module. The constructor should take input and output dimensions, and optional parameters like bias, device, and dtype. The user specified that the test uses bias=False, so maybe the TritonLinear should also include a bias parameter but default to False to match the torch module.

Next, the forward method. For the forward pass, the input is a tensor of shape (B, I), and the weight is (O, I). The output should be (B, O). The Triton kernel for matrix multiplication needs to be implemented here. Autotuning requires specifying configs with different block sizes and maybe group sizes for tiling. Using triton.autotune would involve defining several configs and letting Triton choose the best one based on the input sizes.

Similarly, the backward pass involves computing gradients for the input and weight. The backward kernel will need to handle the derivatives. The backward function in Triton typically requires writing two separate kernels: one for the gradient of the input (dx) and one for the gradient of the weight (dw). Each of these will also need autotuned configurations.

Wait, the user mentioned using super blocks. I think super blocking is a technique where the computation is tiled into larger blocks to improve memory access patterns. So the Triton kernels should use multiple levels of tiling, perhaps with GROUP_SIZE for the super blocks. The autotune parameters should include these group sizes as well.

In the forward kernel, the input is (B, I), weight is (O, I), output is (B, O). The matrix multiplication is input @ weight.T. The Triton kernel will need to tile the computation across B, I, and O dimensions. The block sizes for B, I, O (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) should be autotuned. Similarly, in the backward kernels, similar tiling would be needed for the gradients.

For the autotune configs, each config will specify values for these block sizes and group sizes. For example, in the forward kernel, possible block sizes could be 16, 32, 64, etc. The group size might control how many blocks are processed together in a super block.

The Triton kernels should use the @triton.autotune decorator with a list of configs. Each config is a triton.Config object that sets the block sizes and other parameters. The kwargs in the config can be used to pass these values to the kernel.

Then, in the forward and backward methods, after computing the output, we need to handle the backward pass by defining a backward function that computes the gradients. PyTorch's automatic differentiation won't work here, so we need to manually define the backward pass using custom autograd Functions.

Wait, right. To integrate Triton kernels with PyTorch's autograd, I need to create a custom autograd Function that implements both forward and backward using Triton kernels. The TritonLinear module's forward method would then use this Function.

So the structure would be:

- TritonLinear is a nn.Module that holds the weight and bias parameters.
- The forward method calls a custom autograd Function (maybe called _TritonLinearFunction) which contains the Triton kernels for both forward and backward.
- The _TritonLinearFunction's forward method uses the forward Triton kernel, and its backward method uses the backward kernels for gradients.

In the custom Function's forward, we save the input and weight for use in the backward pass. Then, in backward, we compute d_input and d_weight using the saved tensors and the output gradient.

Now, writing the Triton kernels. For the forward pass:

The kernel will take input (B, I), weight (O, I), and produce output (B, O). Each element in the output is a dot product of a row in input and a row in weight (since weight is O x I, so rows are the output features).

The kernel can be structured with two-dimensional blocks, where each block computes a tile of the output matrix. For example, block size M for rows (B dimension) and N for columns (O dimension). Each block would load a tile of input and weight, compute the partial sum, and accumulate.

But since the inner dimension is I, we need to loop over I in chunks. Using a block size K for the reduction dimension. So the kernel would have three levels of tiling: M (B), N (O), K (I). For each block, it loads a block of input (M x K) and weight (N x K), computes a matrix product of (M x K) and (K x N) to get (M x N), and accumulates into the output.

Autotuning would involve varying M, N, K, and perhaps group sizes for super blocks. The group size might determine how many K blocks are processed together before moving to the next, improving cache utilization.

Similarly, the backward pass for d_input is computed as d_output @ weight, which is a matrix multiplication of (B, O) and (O, I) resulting in (B, I). The kernel for d_input would have similar tiling but transposed weight.

The backward pass for d_weight is input.T @ d_output, which is (I, B) multiplied by (B, O) resulting in (I, O). However, in PyTorch, the gradient for weight is actually the sum over the batch dimension, so it's a (I, O) tensor. The kernel for d_weight would need to accumulate across the B dimension, which might complicate the tiling. Maybe using atomic operations to accumulate the partial sums from different blocks.

Each of these kernels (forward, d_input, d_weight) would need their own Triton implementations with autotuned parameters.

Implementing all of this requires writing three Triton kernels: forward, backward for input, and backward for weight.

Now, considering the autotune configurations. For the forward kernel, possible block sizes could be combinations like BLOCK_SIZE_B=64, BLOCK_SIZE_O=64, and BLOCK_SIZE_I=64, and different group sizes. But the exact values would need to be determined through autotuning.

In the code, the @triton.autotune decorator is used with a list of configs. For example:

configs = [
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=4),
    ...
]

Each config specifies the block sizes and other parameters like num_warps.

Putting this together, the TritonLinear module will have:

- An __init__ method that initializes weight (and optionally bias) as a Parameter.
- A forward method that applies the _TritonLinearFunction, passing the input, weight, and bias (if any).

The _TritonLinearFunction's forward and backward methods will call the respective Triton kernels.

But the test script provided does not include bias, so maybe the TritonLinear should have a bias parameter but default to False. However, in the test's module_generator, the torch module is initialized with bias=False. So the TritonLinear should mirror that.

Now, the code structure:

In linear_triton.py, define the TritonLinear class, and the custom autograd Function.

The forward kernel:

Need to write a triton.jit function that performs the matrix multiplication. Let's call it _forward_kernel. It will take pointers to input, weight, output, and relevant dimensions (B, I, O). The kernel will use 2D grids, with each block handling a portion of B and O.

The backward kernels:

One for d_input (matrix multiply d_output and weight), and another for d_weight (matrix multiply input and d_output, but summed over B). The d_weight kernel is more complex due to the reduction over B.

But in PyTorch, the gradient for the weight in a linear layer is input.T @ d_output, which is a (I, O) matrix. So the kernel for d_weight needs to compute this product efficiently, possibly by tiling across I and O, and parallelizing over B.

For autotuning in the backward kernels, similar block sizes would be used, perhaps with different configurations optimized for the specific operations.

Considering memory layout, the input, weight, and output are all in row-major format. The kernels should be designed to maximize contiguous memory access.

Another point is handling different data types. The test uses torch.float16, so the kernels should specify that they work with fp16. The Triton code should use tl.float16 for the input, weight, and output pointers.

Additionally, in the test code, the input is normalized along the last dimension. But the Triton code doesn't need to handle that; it's part of the input generation.

Now, the custom autograd Function:

class _TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        # save input and weight for backward
        ctx.save_for_backward(input, weight)
        # compute output using triton kernel
        output = ... # call the forward kernel
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # compute gradients using triton backward kernels
        grad_input = ... # call d_input kernel
        grad_weight = ... # call d_weight kernel
        return grad_input, grad_weight, None  # None for bias if not present

But wait, the TritonLinear class may have a bias, but in the test, it's not used. So for now, ignore the bias. The code can be extended later, but the problem states to implement nn.linear, which includes bias as an option. But the test uses bias=False, so the initial code can omit bias handling to simplify.

In the TritonLinear __init__, initialize the weight as a Parameter. For example:

self.weight = nn.Parameter(torch.empty((output_features, input_features), dtype=dtype, device=device))
nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

But how is the weight initialized? The PyTorch Linear layer initializes weight and bias with kaiming uniform. So to match, the TritonLinear should do the same.

Putting it all together:

The linear_triton.py will have:

- The _forward_kernel, _backward_dinput_kernel, _backward_dweight_kernel functions decorated with @triton.jit and @triton.autotune.
- The _TritonLinearFunction that uses these kernels in forward and backward.
- The TritonLinear class that uses this Function in its forward method.

Now, the autotune configurations for the forward kernel. For example:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128}, num_warps=8),
        # more configs...
    ],
    key=['B', 'I', 'O']
)

The key parameters are the dimensions B, I, O, which will determine which config is selected.

Similarly, the backward kernels will have their own autotune configs.

For the backward d_input kernel, which computes grad_input = grad_output @ weight, the kernel will have block sizes for B and I dimensions.

For the backward d_weight kernel, which computes grad_weight = input.T @ grad_output, the kernel will need to tile the I and O dimensions and sum over B. This might require using atomic operations to accumulate the partial sums across different blocks.

But in Triton, atomic operations are supported for certain data types. Since we're using float16, but accumulation might be in float32, perhaps we need to cast to float32 for the accumulators.

Wait, the weight gradient is input.T @ grad_output. So for each element (i, o) in grad_weight, it's the sum over b of input[b, i] * grad_output[b, o]. So the kernel for d_weight needs to compute this sum across all B.

This is a reduction over the B dimension, which can be tricky. One approach is to partition the B dimension into chunks, compute partial sums for each chunk, and then sum them up. However, in Triton, each block can compute a partial sum and then use atomic addition to accumulate into the output tensor.

Alternatively, for large B, the kernel can use tiling in the B dimension and accumulate partial sums in shared memory before writing to the output with atomic operations.

This adds complexity to the d_weight kernel. So the d_weight kernel would need to handle tiling in I and O dimensions, as well as B for reduction.

Given the complexity, perhaps the d_weight kernel should use a block size for I and O, and each block is responsible for a tile of (I_tile, O_tile), and then iterates over the B dimension in chunks, accumulating the sum. Each thread block processes a range of B, and then atomic adds to the grad_weight tensor.

But in Triton, atomic operations are available for certain data types. For example, tl.atomic_add can be used on float32 tensors. Since the grad_weight is of dtype float16 (as per the test), but accumulation in float32 is better for precision, perhaps the kernel should cast to float32 for accumulation and then store back to float16.

Alternatively, the grad_weight tensor can be in float32, but the user's code uses float16. Hmm. The test uses dtype=torch.float16, so the grad_weight should also be float16. But during computation, using float32 for accumulation would help prevent precision loss.

This requires handling the data types carefully. For example, in the kernel, the input and grad_output are loaded as float16, then cast to float32 for computation, accumulated, and then stored as float16.

So the d_weight kernel would look something like:

for b_idx in range(0, B, BLOCK_SIZE_B):
    # load input[b_idx:b_idx+BLOCK_SIZE_B, i] and grad_output[b_idx:..., o]
    # compute the partial sum for this block
    # accumulate into a local variable
# after all blocks, atomic add the sum to grad_weight[i, o]

But how to handle this in Triton. Maybe each program ID is responsible for a specific (i, o) tile, and iterates over B in steps, accumulating the sum. Then, after processing all B steps, the accumulated sum is stored to grad_weight[i, o] using an atomic add.

This approach would require the kernel to have 2D grid dimensions for I and O, and each block handles a tile of (BLOCK_SIZE_I, BLOCK_SIZE_O). Then, for each element in the tile, the thread iterates over B in chunks of BLOCK_SIZE_B, loads the corresponding input and grad_output elements, multiplies them, and accumulates the sum. Finally, the accumulated sum is added to the grad_weight using atomic operations.

This is a feasible approach. However, atomic operations can introduce some overhead, but they are necessary for correctness when multiple blocks contribute to the same (i, o) element.

Now, writing the code for the d_weight kernel:

@triton.jit
def _backward_dweight_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_grad_output_b, stride_grad_output_o,
    stride_grad_weight_i, stride_grad_weight_o,
    GROUP_SIZE_B: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_B: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_o = tl.program_id(1)

    # Define the offsets for the block in I and O dimensions
    i_offset = pid_i * BLOCK_SIZE_I
    o_offset = pid_o * BLOCK_SIZE_O

    # Create masks to prevent out-of-bounds access
    i_mask = i_offset + tl.arange(0, BLOCK_SIZE_I) < I
    o_mask = o_offset + tl.arange(0, BLOCK_SIZE_O) < O

    # Initialize accumulator to zero
    accumulator = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)

    # Loop over B in chunks of BLOCK_SIZE_B * GROUP_SIZE_B
    for b in range(0, B, BLOCK_SIZE_B * GROUP_SIZE_B):
        # Loop within the group
        for g in range(GROUP_SIZE_B):
            b_current = b + g * BLOCK_SIZE_B
            # Check if current B block is within bounds
            b_mask = b_current + tl.arange(0, BLOCK_SIZE_B) < B
            # Load input and grad_output for current B block
            input_block = tl.load(input_ptr + b_current * stride_input_b + i_offset[:, None] * stride_input_i,
                                  mask=i_mask[:, None] & b_mask[None, :], other=0.0)
            grad_output_block = tl.load(grad_output_ptr + b_current * stride_grad_output_b + o_offset[None, :] * stride_grad_output_o,
                                        mask=b_mask[:, None] & o_mask[None, :], other=0.0)
            # Convert to float32 for computation
            input_block = input_block.to(tl.float32)
            grad_output_block = grad_output_block.to(tl.float32)
            # Compute outer product and accumulate
            accumulator += tl.dot(input_block, grad_output_block, allow_tf32=True)

    # Convert accumulator to fp16 and atomically add to grad_weight
    accumulator = accumulator.to(tl.float16)
    grad_weight_ptr_i = grad_weight_ptr + i_offset * stride_grad_weight_i + o_offset * stride_grad_weight_o
    tl.atomic_add(grad_weight_ptr_i + (tl.arange(0, BLOCK_SIZE_I)[:, None] * stride_grad_weight_i +
                                       tl.arange(0, BLOCK_SIZE_O)[None, :] * stride_grad_weight_o),
                  accumulator, mask=i_mask[:, None] & o_mask[None, :])

This kernel uses a group size for B, allowing super blocks to process multiple B blocks at once, improving memory locality. The autotune parameters include BLOCK_SIZE_I, BLOCK_SIZE_O, BLOCK_SIZE_B, and GROUP_SIZE_B.

But integrating all of this into the code requires careful handling of the pointers, strides, and masks to ensure correctness.

Once the kernels are written, the _TritonLinearFunction's backward method would call both the d_input and d_weight kernels.

Putting this all together into code is quite complex, and there are many potential areas for errors, such as incorrect strides, misaligned blocks, or improper use of atomic operations.

Testing the code using the provided benchmark functions is essential. The test code in the __main__ block will generate inputs of various sizes, check correctness against the PyTorch implementation, and benchmark performance.

Potential issues to consider:

1. Correctness of the kernel implementations, especially edge cases where dimensions are not multiples of block sizes.
2. Proper handling of data types and accumulation in higher precision to prevent overflow or underflow.
3. Efficient use of shared memory and registers to maximize performance.
4. Correct configuration of autotune parameters to cover a variety of possible input sizes.

Given the complexity, the code will need thorough testing and validation, but the benchmark.py functions provided should help in verifying correctness and performance.
</think>

```python
# filename: linear_triton.py
import torch
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Compute matrix multiplication between input (B, I) and weight (O, I) to produce output (B, O).

    Args:
        input_ptr (triton.Pointer): Pointer to input tensor of shape (B, I)
        weight_ptr (triton.Pointer): Pointer to weight tensor of shape (O, I)
        output_ptr (triton.Pointer): Pointer to output tensor of shape (B, O)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        strides: Tensor stride information for memory access patterns
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for super-block optimization in batch dimension

    Explanation:
        1. Calculate program IDs for 2D grid (B/O dimensions)
        2. Load input and weight blocks using super-block grouping
        3. Perform matrix multiplication with accumulation
        4. Store results to output tensor
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B * GROUP_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for _ in range(0, I, BLOCK_SIZE_I):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * stride_weight_i
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    tl.store(output_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _backward_dinput_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    stride_grad_output_b, stride_grad_output_o,
    stride_weight_o, stride_weight_i,
    stride_grad_input_b, stride_grad_input_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Compute gradient for input tensor using matrix multiplication between grad_output (B, O) and weight (O, I).

    Args:
        grad_output_ptr (triton.Pointer): Pointer to gradient output tensor of shape (B, O)
        weight_ptr (triton.Pointer): Pointer to weight tensor of shape (O, I)
        grad_input_ptr (triton.Pointer): Pointer to gradient input tensor of shape (B, I)
        B (int): Batch dimension size
        O (int): Output feature dimension size
        I (int): Input feature dimension size
        strides: Tensor stride information for memory access patterns
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for super-block optimization in batch dimension

    Explanation:
        1. Calculate program IDs for 2D grid (B/I dimensions)
        2. Load grad_output and weight blocks using super-block grouping
        3. Perform matrix multiplication with accumulation
        4. Store results to grad_input tensor
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, O)
    
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B * GROUP_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for _ in range(0, O, BLOCK_SIZE_I):
        grad_output_block = tl.load(grad_output_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output_block, weight_block, allow_tf32=True)
        grad_output_ptrs += BLOCK_SIZE_I * stride_grad_output_o
        weight_ptrs += BLOCK_SIZE_I * stride_weight_i
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    grad_input_ptrs = grad_input_ptr + offs_b[:, None] * stride_grad_input_b + offs_i[None, :] * stride_grad_input_i
    tl.store(grad_input_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1}, num_warps=8),
    ],
    key=['I', 'O', 'B']
)
@triton.jit
def _backward_dweight_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_grad_output_b, stride_grad_output_o,
    stride_grad_weight_i, stride_grad_weight_o,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Compute gradient for weight tensor using matrix multiplication between input (B, I) and grad_output (B, O).

    Args:
        input_ptr (triton.Pointer): Pointer to input tensor of shape (B, I)
        grad_output_ptr (triton.Pointer): Pointer to gradient output tensor of shape (B, O)
        grad_weight_ptr (triton.Pointer): Pointer to gradient weight tensor of shape (I, O)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        strides: Tensor stride information for memory access patterns
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for super-block optimization in batch dimension

    Explanation:
        1. Calculate program IDs for 2D grid (I/O dimensions)
        2. Load input and grad_output blocks using super-block grouping
        3. Perform outer product accumulation across batch dimension
        4. Atomically add results to grad_weight tensor
    """
    pid_i = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_b = tl.arange(0, B // (GROUP_SIZE_B * BLOCK_SIZE_O)) * (GROUP_SIZE_B * BLOCK_SIZE_O)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)
    
    for _ in range(0, B, BLOCK_SIZE_O * GROUP_SIZE_B):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I), other=0.0)
        grad_output_block = tl.load(grad_output_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O), other=0.0)
        acc += tl.dot(input_block.T, grad_output_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_input_b
        grad_output_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_grad_output_b
    
    grad_weight_ptr += offs_i[:, None] * stride_grad_weight_i + offs_o[None, :] * stride_grad_weight_o
    tl.atomic_add(grad_weight_ptr, acc, mask=(offs_i[:, None] < I) & (offs_o[None, :] < O))

class _TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = (
            triton.cdiv(B, 256 * 4),  # Using maximum super-block grouping
            triton.cdiv(O, 64),
        )
        
        _forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            GROUP_SIZE_B=4
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]
        
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)
        
        # Compute grad_input
        grid_input = (
            triton.cdiv(B, 256 * 4),
            triton.cdiv(I, 64),
        )
        _backward_dinput_kernel[grid_input](
            grad_output, weight, grad_input,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            GROUP_SIZE_B=4
        )
        
        # Compute grad_weight
        grid_weight = (
            triton.cdiv(I, 64),
            triton.cdiv(O, 64),
        )
        _backward_dweight_kernel[grid_weight](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
            GROUP_SIZE_B=4
        )
        
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return _TritonLinearFunction.apply(x, self.weight)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        return triton_module, torch_module

    def forward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        return input

    from benchmark import verify_correctness_forward, benchmark_performance_forward
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
    benchmark_performance_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])

    def backward_generator(B, I, O, device='cuda', dtype=torch.float16):
        input = torch.randn(B, I, device=device, dtype=dtype)
        input = torch.nn.functional.normalize(input, dim=-1)
        d_output = torch.randn(B, O, device=device, dtype=dtype)
        d_output = torch.nn.functional.normalize(d_output, dim=-1)
        return input, d_output

    from benchmark import verify_correctness_backward, benchmark_performance_backward
    verify_correctness_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
    benchmark_performance_backward(module_generator, backward_generator, [B_sizes, I_sizes, O_sizes])
```

<test>
To test the TritonLinear implementation, run the following command. Ensure benchmark.py and _benchmark_util.py are present in the same directory.

```bash
python linear_triton.py
```
This will execute correctness checks and performance benchmarks for both forward and backward passes across various tensor sizes. Results will be saved as JSON files and visualized in PNG graphs.
</test>