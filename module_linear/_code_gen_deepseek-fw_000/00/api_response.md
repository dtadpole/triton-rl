Timestamp: 20250315_191539

<think>
Okay, I need to create a Triton-based implementation of PyTorch's nn.Linear module, including both forward and backward passes. Let me start by understanding the requirements.

First, the user specified using autotune for block sizes in both forward and backward kernels. The block sizes are BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. Also, they mentioned super blocks and GROUP_SIZE_* parameters for autotuning. Triton's autotune functionality will help here. I remember that Triton allows using configs with different parameters, and the autotuner selects the best ones based on performance.

For the forward pass, the linear layer's operation is matrix multiplication between input (B, I) and weight (O, I), resulting in output (B, O). The Triton kernel needs to efficiently handle this multiplication. The input is a 2D tensor, so the kernel should be designed to process blocks of the input and weight matrices. Using block pointers and proper memory coalescing will be important for performance.

In the backward pass, we need to compute gradients for the input (dx) and weight (dw). The gradients are calculated using the chain rule. For dx, it's a matrix multiplication between d_output (B, O) and weight (O, I), resulting in (B, I). For dw, it's a matrix multiplication between input (I, B) (transposed) and d_output (B, O), resulting in (O, I). These operations also need efficient Triton kernels with autotuned block sizes.

The code structure should include a TritonLinear class that inherits from torch.nn.Module. This class will initialize the weight parameter and define the forward method. The forward method will call a Triton kernel, and the backward method will use custom autograd functions (TritonLinearFwd, TritonLinearBwd) to handle the gradient computations.

I need to define two Triton kernels: one for the forward pass (linear_forward_kernel) and one for the backward pass (linear_backward_kernel). Each kernel will have autotuned configurations for block sizes and group sizes. The autotuner will test different combinations to find the fastest configuration.

In the forward kernel, each block will handle a portion of the output matrix. The block dimensions will be determined by BLOCK_SIZE_B and BLOCK_SIZE_O. Similarly, the backward kernel will have blocks handling parts of the gradient computations. Using super blocks might involve processing multiple elements per thread to optimize memory access and computation.

I should also ensure that the weight matrix is stored in a transposed manner if necessary, as PyTorch's linear layer uses weight (O, I), which is different from the typical (I, O) shape. Wait, no—PyTorch's nn.Linear uses weight shape (out_features, in_features), which is correct for the multiplication input @ weight.T. So in Triton, when accessing the weight, we might need to transpose it, but maybe the kernel can handle it by accessing columns as rows.

Wait, the standard matrix multiplication for a linear layer is input (B, I) multiplied by weight (I, O), resulting in (B, O). But PyTorch's weight is stored as (O, I), so the actual operation is input @ weight.T. So in the Triton kernel, when we access the weight, we need to read it as (I, O) for the multiplication. Hmm, this might affect how the kernel is structured. Alternatively, the weight can be stored transposed in the Triton module. Let me check.

The TritonLinear class initializes the weight as a parameter with shape (output_features, input_features), same as PyTorch. So in the forward pass, the kernel needs to compute input @ weight.T. Therefore, the kernel should access the weight in a transposed manner. Alternatively, maybe the kernel can be written to handle the transpose implicitly.

In the forward kernel, each element in the output is a dot product of a row from input and a row from weight (since weight is (O, I), so rows are of size I). So, for output[b, o], it's sum over i of input[b, i] * weight[o, i]. Therefore, in the kernel, when loading the weight, we can treat it as a 2D tensor of (O, I), and each output element uses a row from weight.

The backward pass for the input gradient (dx) is d_output @ weight. For the weight gradient (dw), it's input.T @ d_output. These operations need to be implemented efficiently in Triton.

Now, for the kernels. Let's outline the forward kernel first.

The forward kernel will take pointers to input, weight, and output. The input is (B, I), weight is (O, I), output is (B, O). Each block should compute a tile of the output. The block size for B (BLOCK_SIZE_B) and O (BLOCK_SIZE_O) can be autotuned. The kernel will iterate over the I dimension, loading blocks of input and weight, accumulating the sum into the output tile.

Similarly, the backward kernel for dx (input gradient) will multiply d_output (B, O) with weight (O, I), resulting in (B, I). The kernel for dw (weight gradient) will multiply input.T (I, B) with d_output (B, O), resulting in (I, O), but since the weight gradient should be (O, I), we need to transpose it. Wait, no: the gradient for weight is input.T @ d_output, which gives (I, B) @ (B, O) = (I, O), but the weight is (O, I), so the gradient should be (O, I), which is the transpose of (I, O). Therefore, the dw kernel needs to compute the outer product and accumulate gradients accordingly.

But the user's test code for backward pass expects dx and dw, so the TritonLinear's backward function should return both. Therefore, in the backward pass, we need two separate kernels: one for computing dx and another for dw. Alternatively, a single kernel that handles both, but that might complicate things. Probably better to split into two kernels.

Wait, the autograd function's backward method needs to compute gradients for input and weight. So in the TritonLinearBwd class's backward method, we'll have to call two kernels: one for dx and one for dw.

Each of these backward operations (dx and dw) will have their own Triton kernels with autotuned parameters.

Now, considering memory access patterns. For the forward pass, each output element is a dot product of a row from input and a row from weight. Using shared memory to cache tiles of input and weight can improve performance. The kernel should be designed to maximize coalesced memory accesses and reuse cached data.

For autotuning, the configs for the forward kernel might include different BLOCK_SIZE_B, BLOCK_SIZE_O, and GROUP_SIZE_I (the number of elements along I processed per iteration). Similarly, the backward kernels will have their own block sizes and group sizes.

In code, each kernel will be decorated with @triton.autotune, providing a list of configs. For example, for the forward kernel:

configs = [
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 32}, num_warps=4),
    ...
]

The same approach applies to the backward kernels.

Now, the TritonLinear module's forward method will call the forward kernel via the autograd function. The autograd function (TritonLinearFwd) saves the input and weight for the backward pass.

In the backward pass, the TritonLinearBwd function will receive d_output and compute dx and dw using the saved input and weight. Each gradient computation (dx and dw) will use their respective Triton kernels with autotuning.

Testing the code involves comparing the outputs and gradients with PyTorch's native implementation. The provided benchmark code uses verify_correctness_forward and verify_correctness_backward functions from benchmark.py. The TritonLinear module must match the PyTorch module's outputs and gradients within a tolerance.

Potential issues to consider:

1. Data types: The code uses torch.float16. Ensure that the Triton kernels handle fp16 correctly, possibly using tl.dot with appropriate ACC_TYPE.
2. Autotuning parameters: Choosing the right block sizes and group sizes for different matrix dimensions is crucial for performance. Testing various combinations is necessary.
3. Correctness in matrix dimensions: Ensuring that the kernels compute the correct elements, especially when dealing with transposed weight matrices.
4. Gradient accumulation: In the backward kernels, correctly summing the contributions to dw from different blocks.

Now, structuring the code:

- The TritonLinear class initializes the weight parameter.
- The forward method uses TritonLinearFwd.apply, which calls the forward kernel.
- The forward kernel is implemented with autotuning.
- The backward method (in TritonLinearBwd) uses two kernels for dx and dw, each with autotuning.

Let me draft the code step by step.

First, the TritonLinear class:

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # initialize weight
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return TritonLinearFwd.apply(x, self.weight)

Then, the autograd functions:

class TritonLinearFwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        output = ...  # call the forward kernel
        return output

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        dx, dw = TritonLinearBwd.apply(x, weight, d_output)
        return dx, dw

Wait, perhaps the backward function should directly compute the gradients. Alternatively, have a separate autograd function for the backward pass. Alternatively, in the backward method of TritonLinearFwd, we can directly call the backward kernels.

Wait, the backward pass for the linear layer involves computing dx = d_output @ weight, and dw = x.T @ d_output. So, in the backward method of TritonLinearFwd, when given d_output, we need to compute dx and dw. These can be computed using Triton kernels.

So, in the TritonLinearFwd's backward method:

class TritonLinearFwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        # call forward kernel and return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input = ...  # compute using backward kernel for dx
        grad_weight = ...  # compute using backward kernel for dw
        return grad_input, grad_weight

Then, the kernels for dx and dw are called here.

But implementing this requires writing two separate backward kernels. Alternatively, a single function that computes both gradients.

But for autotuning, each kernel (dx and dw) would have their own autotuned parameters. So, two separate kernels.

Let me outline the forward kernel:

@triton.autotune(...)
@triton.jit
def linear_forward_kernel(...):
    # kernel code

Similarly, the backward kernels:

@triton.autotune(...)
@triton.jit
def linear_backward_input_kernel(...):
    # computes dx = grad_output @ weight

@triton.autotune(...)
@triton.jit
def linear_backward_weight_kernel(...):
    # computes dw = x.T @ grad_output

In the backward pass of TritonLinearFwd, these kernels are called.

Now, implementing these kernels.

Starting with the forward kernel. The input is x (B, I), weight is (O, I). Output is (B, O). Each element is sum over i of x[b, i] * weight[o, i].

The kernel can be structured with each program handling a block of B and O. For example, pid splits the B and O dimensions into blocks. For each block, compute the sum across I.

So, the block's B and O indices are determined by the program ID. Then, for each element in the block, accumulate the sum over I.

Using tl.dot can help here. For example, loading a block of x and weight, then using tl.dot to compute the partial sums.

But I need to think about how to tile the I dimension. The GROUP_SIZE_I could be the number of elements along I processed in each iteration.

Alternatively, the kernel can use a loop over I in chunks of BLOCK_SIZE_I, loading tiles from x and weight, and accumulating the sum.

Wait, the forward pass is a matrix multiplication. A standard approach in Triton is to tile the matrices into blocks and compute the product using tiling and accumulation.

For example, each block computes a tile of the output matrix by iterating over the K dimension (I in this case) and accumulating the product of tiles from x and weight.

But since x is (B, I) and weight is (O, I), the output is (B, O), which is equivalent to x @ weight.T. So, it's a standard matrix multiplication where the second matrix is transposed.

Triton's matrix multiplication example can be adapted here. The example from Triton's documentation uses tiling for M, N, and K dimensions.

In our case, M is B, N is O, and K is I. So the kernel can be structured similarly to the matrix multiplication example.

The autotune configs would include different BLOCK_SIZE_M (B), BLOCK_SIZE_N (O), and BLOCK_SIZE_K (I), along with GROUP_SIZE parameters.

But the user's problem specifies using BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. So maybe BLOCK_SIZE_B corresponds to M (B), BLOCK_SIZE_O to N (O), and BLOCK_SIZE_I to K (I).

So, the forward kernel's autotune configs would vary these block sizes and group sizes.

Next, the backward pass for dx (input gradient) is a matrix multiplication between grad_output (B, O) and weight (O, I), resulting in (B, I). This is another matrix multiplication, similar to the forward pass but with different dimensions.

The kernel for dx would multiply grad_output (B, O) with weight (O, I), so the output is (B, I). This can be implemented similarly to the forward kernel, transposing the weight if necessary.

Wait, weight is (O, I), so grad_output (B, O) @ weight (O, I) gives (B, I), which is correct. So the kernel for dx would be a matrix multiplication of grad_output (B, O) and weight (O, I).

For the dw (weight gradient), it's x.T (I, B) @ grad_output (B, O), resulting in (I, O), which needs to be transposed to (O, I) since the weight's gradient should have the same shape as the weight (O, I). Therefore, the kernel for dw needs to compute x.T @ grad_output and then transpose the result, or compute the outer product in a way that accumulates into (O, I).

Alternatively, each element dw[o, i] is the sum over b of x[b, i] * grad_output[b, o]. So the kernel can accumulate these sums across the B dimension. This is a batched outer product between x's columns and grad_output's columns.

Implementing this in Triton requires a kernel that efficiently computes these outer products. Each block could handle a tile of (O, I), and for each element in the tile, accumulate the sum over B.

This suggests that the dw kernel will have a different structure compared to the forward and dx kernels, as it's reducing along the B dimension.

So, for the dw kernel, block sizes might be BLOCK_SIZE_O and BLOCK_SIZE_I, and each thread computes the sum over B for a particular (o, i) pair.

But how to efficiently parallelize this? Since B could be large, we can split B into chunks and have each block process a chunk for a tile of (o, i). Then, atomic additions might be needed to accumulate the partial sums from different blocks. However, Triton does not support atomic operations on global memory in all cases, so this could be a problem.

Alternatively, using a reduction across B in shared memory within a block. For each (o, i) in the tile, accumulate the sum over a chunk of B, and then write the partial sums to global memory. Then, another kernel could sum these partial results. But this approach complicates the implementation.

Another option is to use a block-wise approach where each block handles a tile of O and I, and each thread in the block iterates over the B dimension, loading x[b, i] and grad_output[b, o], multiplying them, and accumulating the sum. This might be efficient if B is large and can be processed in chunks.

But for large B, this could lead to high register pressure. So, choosing appropriate block sizes and tiling along B is essential.

This suggests that the dw kernel's autotune parameters would include BLOCK_SIZE_B (the chunk of B processed per iteration) and BLOCK_SIZE_O and BLOCK_SIZE_I for the tile size.

Alternatively, perhaps it's more efficient to split O and I into blocks and compute the sum for each (o, i) by iterating over B in parallel.

Overall, implementing the dw kernel is more challenging due to the reduction over B.

Now, putting this all together, I'll need to write three kernels: forward, dx backward, and dw backward.

Each kernel will use autotuned parameters for block sizes and group sizes. The kernels must handle different input sizes specified in B_sizes, I_sizes, O_sizes.

Potential pitfalls:

- Incorrect indexing leading to out-of-bounds access.
- Not handling cases where dimensions are not perfectly divisible by block sizes.
- Data type mismatches, especially with fp16.
- Incorrectly transposing matrices, leading to wrong results.

Testing each kernel's correctness is crucial. The provided benchmark code uses verify_correctness_* functions which compare against PyTorch's implementation. Any discrepancies in the output or gradients must be addressed.

Finally, integrating all these components into the TritonLinear module, ensuring that the autograd functions correctly save and retrieve the necessary tensors for the backward pass.

Now, time to code this step by step, ensuring that each part is correctly implemented with the required Triton kernels and autotuning configurations.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl
import math

# Forward kernel with autotune for block sizes and group sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_I': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_I': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    Computes forward pass of linear layer using matrix multiplication.
    
    Args:
        input_ptr (tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tensor): Pointer to output tensor of shape (B, O)
        B, I, O (int): Dimensions of input and weight tensors
        stride_*: Stride information for tensor access
        BLOCK_SIZE_B, BLOCK_SIZE_O (tl.constexpr): Tile sizes for B and O dimensions
        GROUP_SIZE_I (tl.constexpr): Group size for splitting I dimension
        
    Returns:
        None (writes results to output_ptr)
        
    Explanation:
        1. Calculate program IDs for batch and output feature blocks
        2. Initialize output accumulator
        3. Loop over input features in groups
        4. Load input and weight tiles using block pointers
        5. Compute partial matrix product with tl.dot
        6. Accumulate results and store final output
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    offs_i = tl.arange(0, GROUP_SIZE_I)
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, GROUP_SIZE_I):
        input = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] + i < I), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] + i < I), other=0.0)
        acc += tl.dot(input, weight, allow_tf32=True)
        input_ptrs += GROUP_SIZE_I * stride_input_i
        weight_ptrs += GROUP_SIZE_I * stride_weight_i
        
    acc = acc.to(tl.float16)
    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    tl.store(output_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

# Backward input kernel with autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_O': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_O': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_input_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, I, O,
    stride_grad_output_b, stride_grad_output_o,
    stride_weight_o, stride_weight_i,
    stride_grad_input_b, stride_grad_input_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr
):
    """
    Computes input gradient (dx) using matrix multiplication of grad_output and weight.
    
    Args:
        grad_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        weight_ptr (tensor): Pointer to weight tensor (O, I)
        grad_input_ptr (tensor): Pointer to input gradient tensor (B, I)
        B, I, O (int): Dimensions of tensors
        stride_*: Stride information for tensor access
        BLOCK_SIZE_B, BLOCK_SIZE_I (tl.constexpr): Tile sizes for B and I dimensions
        GROUP_SIZE_O (tl.constexpr): Group size for splitting O dimension
        
    Returns:
        None (writes results to grad_input_ptr)
        
    Explanation:
        1. Calculate program IDs for batch and input feature blocks
        2. Initialize accumulator
        3. Loop over output features in groups
        4. Load grad_output and weight tiles
        5. Compute partial products with tl.dot
        6. Accumulate and store results
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    offs_o = tl.arange(0, GROUP_SIZE_O)
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, GROUP_SIZE_O):
        grad_output = tl.load(grad_output_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] + o < O), other=0.0)
        weight = tl.load(weight_ptrs, mask=(offs_o[:, None] + o < O) & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output, weight, allow_tf32=True)
        grad_output_ptrs += GROUP_SIZE_O * stride_grad_output_o
        weight_ptrs += GROUP_SIZE_O * stride_weight_o
        
    acc = acc.to(tl.float16)
    grad_input_ptrs = grad_input_ptr + offs_b[:, None] * stride_grad_input_b + offs_i[None, :] * stride_grad_input_i
    tl.store(grad_input_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

# Backward weight kernel with autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_B': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_weight_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_grad_output_b, stride_grad_output_o,
    stride_grad_weight_o, stride_grad_weight_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Computes weight gradient (dw) using outer product of input and grad_output.
    
    Args:
        input_ptr (tensor): Pointer to input tensor (B, I)
        grad_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        grad_weight_ptr (tensor): Pointer to weight gradient tensor (O, I)
        B, I, O (int): Dimensions of tensors
        stride_*: Stride information for tensor access
        BLOCK_SIZE_O, BLOCK_SIZE_I (tl.constexpr): Tile sizes for O and I dimensions
        GROUP_SIZE_B (tl.constexpr): Group size for splitting B dimension
        
    Returns:
        None (writes results to grad_weight_ptr)
        
    Explanation:
        1. Calculate program IDs for output and input feature blocks
        2. Initialize accumulator
        3. Loop over batches in groups
        4. Load input and grad_output tiles
        5. Compute outer product with tl.dot
        6. Accumulate and store results atomically
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    offs_b = tl.arange(0, GROUP_SIZE_B)
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, GROUP_SIZE_B):
        mask = (offs_b[None, :] + b < B) & (offs_i[:, None] < I) & (offs_o[:, None] < O)
        input = tl.load(input_ptrs, mask=mask, other=0.0)
        grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0)
        acc += tl.dot(grad_output, input, allow_tf32=True)
        input_ptrs += GROUP_SIZE_B * stride_input_b
        grad_output_ptrs += GROUP_SIZE_B * stride_grad_output_b
        
    acc = acc.to(tl.float16)
    grad_weight_ptrs = grad_weight_ptr + offs_o[:, None] * stride_grad_weight_o + offs_i[None, :] * stride_grad_weight_i
    tl.atomic_add(grad_weight_ptrs, acc, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I))

class TritonLinearFwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        
        linear_forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1)
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]
        
        # Calculate input gradient
        grad_input = torch.empty_like(input)
        grid_dx = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        linear_backward_input_kernel[grid_dx](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1)
        )
        
        # Calculate weight gradient
        grad_weight = torch.zeros_like(weight)
        grid_dw = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        linear_backward_weight_kernel[grid_dw](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1)
        )
        
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return TritonLinearFwd.apply(x, self.weight)

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
To test the implementation, run the following command. Ensure you have PyTorch and Triton installed.

```bash
python linear_triton.py
```
</test>