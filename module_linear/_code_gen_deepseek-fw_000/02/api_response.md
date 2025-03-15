Timestamp: 20250315_191704

<think>
Alright, I need to generate a Triton implementation of PyTorch's nn.Linear module, including both forward and backward passes. The user provided a specific code structure to follow, including using autotune for block sizes and integrating with existing benchmark functions. Let me break this down step by step.

First, the main file is linear_triton.py. The TritonLinear class should mimic the functionality of torch.nn.Linear. Since the user mentioned using autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both forward and backward kernels, I'll need to set up Triton's autotune configs. Also, super blocks and GROUP_SIZE_* parameters should be considered for autotuning.

Starting with the forward pass. The linear operation is essentially a matrix multiplication between the input and the weight matrix, plus a bias if present. But in the provided benchmark code, the torch module is initialized with bias=False, so maybe the Triton implementation can ignore bias for now. Wait, the user's module_generator creates TritonLinear with bias=False? Let me check. The sample code shows 'bias=False' for torch_module, but the TritonLinear isn't specified. Wait, in the module_generator, TritonLinear is initialized with I, O, device, and dtype, but the code might need to handle bias. The original problem statement says to implement nn.linear, which usually includes a bias option. However, the benchmark's torch module is created with bias=False. To match that, TritonLinear should probably not include a bias. So I'll proceed without bias.

For the forward kernel, the input is of shape (B, I), and the weight is (O, I). The output is (B, O). The matrix multiplication is input @ weight.T. In Triton, I'll need to write a kernel that efficiently handles this. Autotuning block sizes for B, I, O dimensions. The kernel will need to iterate over the input and weight matrices, using blocking to optimize memory access.

Then the backward pass. The backward function needs to compute gradients with respect to the input and the weight. The user's backward generator provides input and d_output. So during backward, given d_output (gradient of loss with respect to output), compute d_input (gradient with respect to input) and d_weight (gradient with respect to weight). The formulas are d_input = d_output @ weight, and d_weight = input.T @ d_output. Both of these are matrix multiplications, which need to be implemented in Triton with autotuned block sizes.

Implementing autotuning requires using Triton's autotune decorator, specifying different configs for block sizes and group sizes. For each kernel (forward and backward), I'll define a set of possible block sizes and use Triton's autotune to select the best ones during the initial runs.

In the forward kernel, the input is (B, I), weight is (O, I). The output is (B, O). The kernel should split the computation into blocks. For example, each block could handle a portion of the B and O dimensions. The autotune parameters would determine how to partition these dimensions for optimal performance.

For the backward kernels, there are two parts: computing d_input and d_weight. The d_input calculation is similar to the forward pass but with weight instead of input. The d_weight calculation is a transpose multiplication, which might benefit from different block sizing. Each of these will need their own Triton kernels with autotune configurations.

Now, considering the structure of the TritonLinear class. It should inherit from torch.nn.Module, and have parameters for weight. The forward method will call the Triton kernel. The backward method would be handled by defining a backward function, but in PyTorch, we usually use custom autograd Functions. So maybe TritonLinear's forward uses a custom autograd Function that implements both forward and backward passes using Triton kernels.

Wait, to properly integrate with PyTorch's autograd system, the TritonLinear module should use a custom autograd.Function. So the forward method of TritonLinear would call a TritonLinearFunction.apply, which includes both the forward and backward Triton implementations.

Yes, that makes sense. So the TritonLinear class will have a weight parameter, and in its forward pass, it uses the custom Function. The Function's forward method calls the Triton kernel, and the backward method calls the backward Triton kernels for gradients.

So the steps are:

1. Define the TritonLinear class, inheriting from nn.Module, initializing the weight matrix.
2. Define a custom autograd Function (TritonLinearFunction) that implements forward and backward using Triton kernels.
3. Implement the forward Triton kernel with autotune.
4. Implement the two backward Triton kernels (for input and weight gradients) with autotune.
5. Integrate these into the TritonLinear module.

Now, writing the Triton kernels. Let's start with the forward kernel. The kernel will take input and weight, and compute output. The block sizes for B, I, O need to be autotuned. For example, BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O could be powers of two, and Triton's autotuner will select the best combination.

In the kernel, each program might handle a block of B and O dimensions. For example, for each block, compute a subset of rows (B) and columns (O) of the output matrix. The kernel will need to load the corresponding blocks of input and weight, perform the matrix multiplication, and store the result.

For the backward pass, the d_input computation is d_output multiplied by the weight matrix. The kernel for this is similar to the forward pass, but the dimensions are different. The weight matrix is (O, I), so d_output (B, O) multiplied by weight (O, I) gives d_input (B, I). So the kernel for d_input would need to iterate over B and I dimensions.

The d_weight computation is input.T (I, B) multiplied by d_output (B, O), resulting in a (I, O) matrix. However, in PyTorch, the weight gradient is accumulated across all batches, so this is a sum over the batch dimension. The kernel for d_weight would need to compute the outer product of input and d_output across the batch dimension, then sum those. This is a bit more complex. The Triton kernel for this would need to efficiently handle the reduction across the B dimension.

Each of these kernels (forward, d_input, d_weight) will have their own autotune configurations. For each, I'll define multiple possible block sizes and group sizes, and let Triton's autotuner select the best during the initial runs.

Now, let's outline the code structure.

First, the TritonLinear class:

import torch
import triton
import triton.language as tl

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
        # Initialize weight similarly to PyTorch's Linear
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

Then, the TritonLinearFunction class using autograd.Function:

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        output = ...  # Call the Triton forward kernel
        return output

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        dx, dw = ...  # Call the Triton backward kernels
        return dx, dw

Now, implementing the forward kernel. Let's call it 'linear_forward_kernel'. The kernel will take pointers to x, weight, and output. The configs for block sizes are autotuned.

@triton.autotune(...)
@triton.jit
def linear_forward_kernel(...):
    ...

Similarly for the backward kernels: 'linear_backward_input_kernel' and 'linear_backward_weight_kernel'.

But how to structure the kernels? Let's think about the forward kernel first.

The input x has shape (B, I), weight is (O, I), output is (B, O). Each element in the output is a dot product of a row in x and a column in weight (since output = x @ weight.T).

In Triton, the kernel can be designed to handle blocks of B and O. For example, each block computes a tile of output of size BLOCK_SIZE_B x BLOCK_SIZE_O. To compute each element in the tile, we need to iterate over the I dimension, accumulating the sum.

So the kernel would have a grid that covers B in blocks of BLOCK_SIZE_B and O in blocks of BLOCK_SIZE_O. For each block, the program ID would determine which part of the output to compute.

The kernel will load a block of x (BLOCK_SIZE_B x BLOCK_SIZE_I) and a block of weight (BLOCK_SIZE_O x BLOCK_SIZE_I), then compute their product, accumulating across the I dimension.

Wait, but the weight is (O, I), so each row in weight corresponds to an output feature. Therefore, when computing x @ weight.T, each row of x (size I) is multiplied by each row of weight (size I), resulting in a scalar for each output feature.

So for the kernel, each block could compute a tile of the output matrix. For example, if BLOCK_SIZE_B is 64 and BLOCK_SIZE_O is 64, each block computes 64x64 elements of the output. But this might not be optimal. Alternatively, each program could handle a row in B and a column in O, but that would be more fine-grained.

Alternatively, using a 2D grid where each block handles a portion of B and O. But I think Triton's examples for matrix multiplication can be adapted here.

Looking at Triton's matrix multiplication example, they have a kernel that tiles both matrices and computes the product in blocks. For our case, since the input is (B, I) and weight is (O, I), the product is (B, O), so each element (b, o) is the dot product of input[b, :] and weight[o, :].

The kernel for this would need to iterate over the I dimension, accumulating the sum. Each block could handle multiple elements in B and O, using tiling.

But for autotuning, the block sizes for B, I, and O dimensions are variables. So the autotuner would try different combinations, like BLOCK_SIZE_B=32, BLOCK_SIZE_O=32, and BLOCK_SIZE_I=64, etc.

The forward kernel would then be autotuned with different configurations. Similarly for the backward kernels.

Now, moving on to the backward pass. The backward for input (dx) is computed as d_output @ weight, which is a (B, O) matrix multiplied by (O, I) matrix, resulting in (B, I). The kernel for this is similar to the forward pass, but the input is d_output and the weight remains the same.

The backward for weight (dw) is computed as x.T @ d_output, which is (I, B) multiplied by (B, O), resulting in (I, O). This is a batched outer product summed over the B dimension. The weight gradient is the sum over all batches, so the kernel needs to accumulate the sum across the B dimension.

The challenge in the weight gradient kernel is handling the reduction over B. This might require a different approach, possibly using atomic operations to accumulate the sum in parallel.

For the weight gradient kernel, each element dw[i, j] is the sum over all b of x[b, i] * d_output[b, j]. So each thread could handle a specific (i, j) pair and iterate over all b in the B dimension. Alternatively, split the B dimension into blocks and perform partial sums, then combine them.

But with Triton's programming model, using block-wise reductions and efficient memory access is key. The kernel could tile the I and O dimensions and compute partial sums for each tile, then aggregate them.

This part might be more complex. The autotuning for the weight gradient kernel would need to consider block sizes for I and O, and perhaps the number of steps along B.

Putting this all together, I'll need to write three Triton kernels: forward, backward_input, and backward_weight. Each with their own autotune configurations.

Now, coding the forward kernel:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 16}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_outputb, stride_outputo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    # Define offsets for B and O dimensions
    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    # Loop over I dimension in blocks
    for i in range(0, I, BLOCK_SIZE_I):
        off_i = i + tl.arange(0, BLOCK_SIZE_I)
        # Load x and weight tiles
        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x_tile = tl.load(x_ptr + off_b[:, None] * stride_xb + off_i[None, :] * stride_xi, mask=x_mask, other=0.0)
        weight_mask = (off_o[:, None] < O) & (off_i[None, :] < I)
        weight_tile = tl.load(weight_ptr + off_o[:, None] * stride_wo + off_i[None, :] * stride_wi, mask=weight_mask, other=0.0)

        # Compute matrix multiplication
        accumulator += tl.dot(x_tile, weight_tile, allow_tf32=True)

    # Write back the result
    output_ptr_ptrs = output_ptr + off_b[:, None] * stride_outputb + off_o[None, :] * stride_outputo
    output_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
    tl.store(output_ptr_ptrs, accumulator.to(tl.float16), mask=output_mask)

But wait, the strides need to be correctly calculated. For x, which is (B, I), the stride for B is I (x.stride(0)), and for I is 1 (x.stride(1)). Weight is (O, I), so stride_wo is weight.stride(0) = I, stride_wi = 1. The output is (B, O), so stride_outputb = O, stride_outputo = 1.

In the kernel, the dot product between x_tile (BLOCK_SIZE_B x BLOCK_SIZE_I) and weight_tile (BLOCK_SIZE_O x BLOCK_SIZE_I) would be a (BLOCK_SIZE_B x BLOCK_SIZE_O) matrix. However, the tl.dot expects the second matrix to be transposed? Wait, in Triton's tl.dot(a, b), it computes a @ b.T if a is (M, K) and b is (N, K), resulting in (M, N). So in this case, x_tile is (BLOCK_SIZE_B, BLOCK_SIZE_I), weight_tile is (BLOCK_SIZE_O, BLOCK_SIZE_I). To compute x_tile @ weight_tile.T, the result is (BLOCK_SIZE_B, BLOCK_SIZE_O), which matches the accumulator.

Yes, so the dot product here is correct.

Now, the grid needs to be launched with enough blocks to cover B and O dimensions. For example:

grid = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(O, BLOCK_SIZE_O))

But since the autotune parameters may vary, the grid is computed based on the selected BLOCK_SIZE_B and BLOCK_SIZE_O.

Now, the forward function:

def linear_forward(x, weight):
    B, I = x.shape
    O, _ = weight.shape
    output = torch.empty((B, O), device=x.device, dtype=x.dtype)
    # Compute grid based on block sizes (will be autotuned)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
    linear_forward_kernel[grid](
        x, weight, output,
        B, I, O,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    return output

But wait, the autotune configs are part of the kernel, so the grid function uses the meta parameters from the autotune config.

This should handle the forward pass.

Now, the backward pass for input (dx = d_output @ weight):

@triton.autotune(...)
@triton.jit
def linear_backward_input_kernel(...):
    # Similar structure to forward kernel, but multiply d_output (B, O) with weight (O, I) to get dx (B, I)
    pass

And the backward pass for weight (dw = x.T @ d_output), which requires summing over the B dimension. This kernel is more complex, as it involves a reduction.

For dw, each element (i, o) is the sum over all b of x[b, i] * d_output[b, o]. This is a batched outer product, summed across batches. The kernel needs to accumulate this sum, which can be done by partitioning the B dimension into blocks and using atomic adds to accumulate the partial sums.

Here's an outline for the backward_weight kernel:

@triton.autotune(...)
@triton.jit
def linear_backward_weight_kernel(
    x_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_o = tl.program_id(1)

    # Offsets for I and O dimensions
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    # Accumulator for dw
    accumulator = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)

    # Loop over B in blocks
    for b in range(0, B, BLOCK_SIZE_B):
        off_b = b + tl.arange(0, BLOCK_SIZE_B)
        # Load x and d_output tiles
        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x_tile = tl.load(x_ptr + off_b[:, None] * stride_xb + off_i[None, :] * stride_xi, mask=x_mask, other=0.0)
        d_output_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
        d_output_tile = tl.load(d_output_ptr + off_b[:, None] * stride_dob + off_o[None, :] * stride_doo, mask=d_output_mask, other=0.0)

        # Compute outer product and accumulate
        accumulator += tl.dot(x_tile.T, d_output_tile, allow_tf32=True)

    # Write accumulator to d_weight
    dw_ptrs = d_weight_ptr + off_i[:, None] * stride_dwi + off_o[None, :] * stride_dwo
    dw_mask = (off_i[:, None] < I) & (off_o[None, :] < O)
    tl.atomic_add(dw_ptrs, accumulator.to(tl.float16), mask=dw_mask)

But handling atomic adds for accumulation in Triton may require careful handling. Also, the kernel would need to split the work across I and O dimensions, with each block handling a tile of I x O and iterating over B in chunks. The GROUP_SIZE parameters may be used to control how I and O are grouped for parallel processing.

Now, integrating these kernels into the custom autograd Function:

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        output = linear_forward(x, weight)
        return output

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        dx = linear_backward_input(d_output, weight)
        dw = linear_backward_weight(x, d_output)
        return dx, dw

But implementing linear_backward_input and linear_backward_weight functions using the Triton kernels.

However, PyTorch expects the backward pass to compute gradients for all inputs of the forward pass. Since the forward pass has inputs x and weight, the backward should return gradients for both, which are dx and dw.

The linear_backward_input function computes dx as d_output @ weight.

The linear_backward_weight function computes dw as x.T @ d_output.

Now, implementing these functions with their respective kernels.

For linear_backward_input:

def linear_backward_input(d_output, weight):
    B, O = d_output.shape
    I = weight.shape[1]
    dx = torch.empty((B, I), device=d_output.device, dtype=d_output.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
    linear_backward_input_kernel[grid](
        d_output, weight, dx,
        B, O, I,
        d_output.stride(0), d_output.stride(1),
        weight.stride(0), weight.stride(1),
        dx.stride(0), dx.stride(1),
    )
    return dx

Similarly, for linear_backward_weight:

def linear_backward_weight(x, d_output):
    B, I = x.shape
    O = d_output.shape[1]
    dw = torch.zeros((I, O), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(I, meta['BLOCK_SIZE_I']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
    linear_backward_weight_kernel[grid](
        x, d_output, dw,
        B, I, O,
        x.stride(0), x.stride(1),
        d_output.stride(0), d_output.stride(1),
        dw.stride(0), dw.stride(1),
    )
    return dw

But wait, the weight gradient should be of shape (O, I), because the original weight is (O, I). However, the computation x.T @ d_output is (I, B) @ (B, O) = (I, O). But the weight gradient should be (O, I). Therefore, we need to transpose the result. Wait, no. The weight is (O, I), so the gradient dw should be (O, I). Hmm, but according to the math, x.T @ d_output is (I, O), which is the gradient for a weight matrix of shape (I, O). But since our weight is (O, I), the gradient would need to be (I, O). But that's not matching. Wait, let's double-check.

The original forward is output = x @ weight.T. Therefore, the gradient of the loss with respect to weight is computed as x.T @ d_output, resulting in (I, B) @ (B, O) = (I, O). However, since the weight is (O, I), the gradient should be (O, I). Therefore, we need to transpose the result of x.T @ d_output to get (O, I). So, dw = (x.T @ d_output).T, which is (O, I).

Oh right! Therefore, the kernel for computing dw should produce (I, O), and then we transpose it to (O, I) before returning. But that would add an extra step. Alternatively, the kernel can directly compute the correct dimensions.

Wait, no. The weight is (O, I), so the gradient dw should have the same shape. The mathematical gradient for weight in the operation output = x @ weight.T is (d_output.T @ x). Because:

If output = x @ W^T, then dL/dW = d_output^T @ x.

So the correct gradient is (d_output.T @ x), which is (O, B) @ (B, I) = (O, I). Therefore, the kernel should compute this instead of x.T @ d_output.

Ah, right. So my earlier approach was wrong. The correct gradient for weight is d_output.T @ x.

So the kernel for dw should compute d_output.T @ x. Which is a (O, B) @ (B, I) = (O, I) matrix multiplication. This is different from what I thought earlier.

Therefore, the backward_weight kernel should handle this. This changes the approach. Now, the kernel needs to compute d_output.T @ x.

So the kernel for weight gradient should take d_output (B, O) and x (B, I), and compute d_output.T (O, B) @ x (B, I) → (O, I). 

This changes the Triton kernel structure. Each element in dw[o, i] is the sum over b of d_output[b, o] * x[b, i]. So the kernel needs to iterate over o and i dimensions, and for each (o, i) pair, sum over all b.

But this is a different access pattern. The kernel could tile the O and I dimensions, and for each tile, accumulate the sum over B.

So, the weight gradient kernel should have a grid of (O, I) blocks. Each block processes a tile of O and I, and iterates over all B, loading slices of d_output and x, multiplying them, and accumulating the sum.

The autotuning parameters would then be BLOCK_SIZE_O and BLOCK_SIZE_I, and GROUP_SIZE parameters for how to partition the B dimension.

This makes the kernel more manageable. Let's outline this:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 16}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_weight_kernel(
    d_output_ptr, x_ptr, d_weight_ptr,
    B, I, O,
    stride_dob, stride_doo,  # d_output strides: (B, O) → strides (O, 1)
    stride_xb, stride_xi,     # x strides: (B, I) → strides (I, 1)
    stride_dwo, stride_dwi,   # d_weight strides: (O, I) → strides (I, 1)
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    for b in range(0, B, BLOCK_SIZE_B):
        off_b = b + tl.arange(0, BLOCK_SIZE_B)
        # Load d_output[off_b, off_o] and x[off_b, off_i]
        d_output_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
        d_output_tile = tl.load(d_output_ptr + off_b[:, None] * stride_dob + off_o[None, :] * stride_doo, mask=d_output_mask, other=0.0)

        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x_tile = tl.load(x_ptr + off_b[:, None] * stride_xb + off_i[None, :] * stride_xi, mask=x_mask, other=0.0)

        # Compute outer product: d_output_tile (BLOCK_SIZE_B, BLOCK_SIZE_O) @ x_tile (BLOCK_SIZE_B, BLOCK_SIZE_I)
        # To compute sum over B, this is equivalent to (d_output_tile.T @ x_tile) → (BLOCK_SIZE_O, BLOCK_SIZE_I)
        # Which is added to the accumulator
        accumulator += tl.dot(d_output_tile.T, x_tile, allow_tf32=True)

    # Write to d_weight[off_o, off_i]
    d_weight_ptr_ptrs = d_weight_ptr + off_o[:, None] * stride_dwo + off_i[None, :] * stride_dwi
    mask = (off_o[:, None] < O) & (off_i[None, :] < I)
    tl.store(d_weight_ptr_ptrs, accumulator.to(tl.float16), mask=mask)

This kernel iterates over blocks of B, and for each block, loads a tile of d_output and x, computes their outer product (transposed d_output tile times x tile), and accumulates the sum into the d_weight matrix.

The grid for this kernel would be (triton.cdiv(O, BLOCK_SIZE_O), triton.cdiv(I, BLOCK_SIZE_I)).

Now, the linear_backward_weight function would be:

def linear_backward_weight(d_output, x):
    O = d_output.shape[1]
    I = x.shape[1]
    B = x.shape[0]
    dw = torch.zeros((O, I), device=d_output.device, dtype=d_output.dtype)
    grid = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
    linear_backward_weight_kernel[grid](
        d_output, x, dw,
        B, I, O,
        d_output.stride(0), d_output.stride(1),
        x.stride(0), x.stride(1),
        dw.stride(0), dw.stride(1),
    )
    return dw

But PyTorch's weight is stored as (O, I), so this dw will have the correct shape.

Now, integrating all this into the TritonLinearFunction:

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        output = linear_forward(x, weight)
        return output

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        # Compute gradients
        dx = linear_backward_input(d_output, weight)
        dw = linear_backward_weight(d_output, x)
        return dx, dw

Wait, no. The backward pass returns gradients for each input of the forward function. The forward function's inputs are x and weight, so backward should return gradients for x and weight. The linear_backward_input computes dx (gradient for x), and linear_backward_weight computes dw (gradient for weight). So the backward method returns (dx, dw).

But in the code above, linear_backward_weight is called with d_output and x, which computes d_output.T @ x → (O, I), which is correct for dw.

Yes.

Now, testing the code with the provided benchmark. The user's main code calls verify_correctness_forward and benchmark_performance_forward, similarly for backward. The TritonLinear should match the behavior of torch.nn.Linear when there's no bias.

Potential issues to consider:

- Autotuning might require multiple runs. The verification and benchmark functions likely handle this by warming up the kernels before timing.
- Numerical precision: Using fp16 could lead to precision differences. The user's benchmark uses atol and rtol of 1e-2, which should accommodate this.
- Correctness of the Triton kernels, especially the backward pass for weight gradient.

Another point: the user's code in the module_generator initializes the TritonLinear and torch.nn.Linear without bias. The TritonLinear as written also doesn't include a bias, so that's correct.

Now, let's write the full code for linear_triton.py, including all kernels and the TritonLinear class.

Putting it all together:

- TritonLinear class with forward using the custom Function.
- The custom Function with forward and backward.
- Triton kernels for forward, backward_input, and backward_weight, each with autotuning.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return linear_forward(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input = linear_backward_input(grad_output, weight)
        grad_weight = linear_backward_weight(grad_output, x)
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    B, I, O,
    x_stride_b, x_stride_i,
    weight_stride_o, weight_stride_i,
    output_stride_b, output_stride_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    for i in range(0, I, BLOCK_SIZE_I):
        off_i = i + tl.arange(0, BLOCK_SIZE_I)
        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x = tl.load(x_ptr + off_b[:, None] * x_stride_b + off_i[None, :] * x_stride_i, mask=x_mask, other=0.0)
        w_mask = (off_o[:, None] < O) & (off_i[None, :] < I)
        weight = tl.load(weight_ptr + off_o[:, None] * weight_stride_o + off_i[None, :] * weight_stride_i, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, weight, allow_tf32=True)

    output_ptrs = output_ptr + off_b[:, None] * output_stride_b + off_o[None, :] * output_stride_o
    output_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=output_mask)

def linear_forward(x, weight):
    B, I = x.shape
    O, _ = weight.shape
    output = torch.empty((B, O), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
    linear_forward_kernel[grid](
        x, weight, output, B, I, O,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1)
    )
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 16}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_input_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    grad_output_stride_b, grad_output_stride_o,
    weight_stride_o, weight_stride_i,
    grad_input_stride_b, grad_input_stride_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for o in range(0, O, BLOCK_SIZE_B):
        off_o = o + tl.arange(0, BLOCK_SIZE_B)
        go_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
        grad_output = tl.load(grad_output_ptr + off_b[:, None] * grad_output_stride_b + off_o[None, :] * grad_output_stride_o, mask=go_mask, other=0.0)
        w_mask = (off_o[None, :] < O) & (off_i[:, None] < I)
        weight = tl.load(weight_ptr + off_o[None, :] * weight_stride_o + off_i[:, None] * weight_stride_i, mask=w_mask, other=0.0)
        accumulator += tl.dot(grad_output, weight, allow_tf32=True)

    grad_input_ptrs = grad_input_ptr + off_b[:, None] * grad_input_stride_b + off_i[None, :] * grad_input_stride_i
    mask = (off_b[:, None] < B) & (off_i[None, :] < I)
    tl.store(grad_input_ptrs, accumulator.to(tl.float16), mask=mask)

def linear_backward_input(grad_output, weight):
    B, O = grad_output.shape
    I = weight.shape[1]
    grad_input = torch.empty((B, I), device=grad_output.device, dtype=grad_output.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
    linear_backward_input_kernel[grid](
        grad_output, weight, grad_input, B, O, I,
        grad_output.stride(0), grad_output.stride(1),
        weight.stride(0), weight.stride(1),
        grad_input.stride(0), grad_input.stride(1)
    )
    return grad_input

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_weight_kernel(
    grad_output_ptr, x_ptr, grad_weight_ptr,
    B, I, O,
    grad_output_stride_b, grad_output_stride_o,
    x_stride_b, x_stride_i,
    grad_weight_stride_o, grad_weight_stride_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    for b in range(0, B, GROUP_SIZE_B):
        off_b = b + tl.arange(0, GROUP_SIZE_B)
        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x = tl.load(x_ptr + off_b[:, None] * x_stride_b + off_i[None, :] * x_stride_i, mask=x_mask, other=0.0)
        go_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
        grad_output = tl.load(grad_output_ptr + off_b[:, None] * grad_output_stride_b + off_o[None, :] * grad_output_stride_o, mask=go_mask, other=0.0)
        accumulator += tl.dot(grad_output.T, x, allow_tf32=True)

    grad_weight_ptrs = grad_weight_ptr + off_o[:, None] * grad_weight_stride_o + off_i[None, :] * grad_weight_stride_i
    mask = (off_o[:, None] < O) & (off_i[None, :] < I)
    tl.atomic_add(grad_weight_ptrs, accumulator.to(tl.float16), mask=mask)

def linear_backward_weight(grad_output, x):
    O = grad_output.shape[1]
    I = x.shape[1]
    grad_weight = torch.zeros((O, I), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
    linear_backward_weight_kernel[grid](
        grad_output, x, grad_weight, B=x.shape[0], I=I, O=O,
        grad_output_stride_b=grad_output.stride(0),
        grad_output_stride_o=grad_output.stride(1),
        x_stride_b=x.stride(0),
        x_stride_i=x.stride(1),
        grad_weight_stride_o=grad_weight.stride(0),
        grad_weight_stride_i=grad_weight.stride(1)
    )
    return grad_weight

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
To test the code, execute the following command:

```bash
python linear_triton.py
```
This will run the correctness checks and performance benchmarks as defined in the `__main__` block. The results will be saved in JSON files (`code.correctness*.json`, `code.performance*.json`) and plots generated for performance comparison.
</test>