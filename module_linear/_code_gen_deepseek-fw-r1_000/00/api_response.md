Timestamp: 20250331_042932

<think>
Okay, I need to create a Triton module for PyTorch's nn.Linear, implementing both forward and backward passes. The user specified using autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both kernels, and using super blocks with autotuned GROUP_SIZE_*. Also, I have to structure the code with detailed comments as per the template provided.

First, let me recall how Triton works. Triton allows writing GPU kernels with a Python-like syntax, which are then JIT-compiled. For matrix operations, tiling is essential to efficiently use GPU memory bandwidth. Autotuning helps find the optimal tile sizes and configurations for different input sizes.

For the forward pass of a linear layer, the operation is a matrix multiplication between the input (B x I) and the weight matrix (I x O), resulting in an output (B x O). The Triton kernel for this would need to handle batches and use tiling for efficient computation.

In the backward pass, we need to compute gradients for the input (dx) and the weights (dw). The gradient for the input is computed by multiplying the upstream gradient (d_output) with the weight matrix transpose. The weight gradient is computed by multiplying the input transpose with the upstream gradient.

Autotuning in Triton is done using the @triton.autotune decorator, which allows specifying configs for different parameters. The configs should explore different block sizes and group sizes for super blocks. Super blocks allow processing multiple blocks together, which can improve memory access patterns.

Now, structuring the TritonLinear class. It should inherit from torch.nn.Module, have a weight parameter, and optionally a bias. The forward method will dispatch to the Triton kernel, and we'll need to define the backward function using a custom autograd Function.

Wait, but how to handle the backward pass? PyTorch's autograd requires defining a backward function that computes gradients. So, the TritonLinear module's forward should use a custom autograd Function that has its own backward method implemented with Triton kernels.

So, the plan is:

1. Create a TritonLinear module that initializes weight (and bias if needed, but the user's test code doesn't include bias).
2. The forward method of TritonLinear uses a custom autograd Function (e.g., _linear_triton) which encapsulates both forward and backward Triton kernels.
3. The _linear_triton Function's forward calls the Triton kernel for matrix multiplication.
4. The backward method of _linear_triton will compute dx and dw using Triton kernels.

Now, the kernels for forward and backward.

Forward Kernel:

The input is (B, I), weight is (O, I) (since in PyTorch, Linear layers are stored as (out_features, in_features)), and output is (B, O).

The kernel needs to tile both the input and weight matrices. Each block will compute a tile of the output. For autotuning, block sizes for B, I, O dimensions need to be explored.

Wait, but in matrix multiplication, the tiling is usually done on the M, N, K dimensions. Here, for the input (B, I) and weight (I, O), the output is (B, O). So, each element in the output is a dot product of a row from input and a column from weight. But for batched matrix multiplication, each batch is an independent matrix multiply.

Alternatively, the kernel can process each batch element in parallel. But that might not be efficient. Alternatively, treat the batch as another dimension to tile.

But maybe it's better to consider the batch dimension as part of the M dimension. So, the input is (B*I) treated as a 2D matrix, and the weight is (I, O). Then, the output would be (B, O). Hmm, perhaps reshaping the input to (B, I) and weight to (I, O), and the output is (B, O).

In Triton, the kernel would iterate over the rows of input (B) and columns of weight (O), computing the dot product over the I dimension.

For tiling, block size along B (BLOCK_SIZE_B), along I (BLOCK_SIZE_I), and O (BLOCK_SIZE_O). But I'm a bit confused about how to structure the tiling here.

Alternatively, for the forward kernel, each program could handle a block of B rows and O columns, accumulating over I. The block sizes for B and O can be autotuned, and the accumulation over I can be split into blocks of size BLOCK_SIZE_I.

The autotune configs would then vary BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O, and GROUP_SIZE_* for super blocks.

Similarly for the backward kernels: dx is input gradient (B, I) computed as d_output (B, O) multiplied by weight (O, I). dw is (O, I) computed as input (I, B) multiplied by d_output (B, O), so dw = input.T @ d_output.

So, the backward pass involves two matrix multiplications: one for dx and one for dw.

Each of these can be implemented as separate Triton kernels, each with their own autotuning configurations.

Now, writing the Triton kernels with the required autotuning.

For the forward kernel:

- The input is of shape (B, I), weight is (O, I), output is (B, O).
- Each block should compute a tile of the output. For example, a tile of size BLOCK_SIZE_B x BLOCK_SIZE_O.
- To compute each element in the tile, we need to accumulate over the I dimension, tiled into BLOCK_SIZE_I chunks.

So, the kernel would have nested loops over the I dimension in blocks of BLOCK_SIZE_I, loading input and weight tiles, and accumulating the products.

The autotune configs would explore different combinations of BLOCK_SIZE_B, BLOCK_SIZE_O, and BLOCK_SIZE_I, along with GROUP_SIZE for super blocks.

Similarly, for the dx kernel (input gradient):

- The d_output is (B, O), weight is (O, I), so the dx is (B, I) = d_output @ weight.T.
- The kernel would multiply d_output (B, O) by weight (I, O) (since weight is stored as O x I, the transpose is I x O).
- Each block computes a tile of B x I.

For the dw kernel (weight gradient):

- input is (B, I), d_output is (B, O), so dw (O, I) is d_output.T @ input.
- Each block computes a tile of O x I.

Each of these operations would have their own Triton kernels with autotuned block sizes.

Now, integrating this into the TritonLinear module.

The module's forward function will take input, pass it through the custom autograd Function (which uses the Triton forward kernel), and return the output.

The custom autograd Function's backward method will use the Triton kernels for dx and dw.

But how to handle the weight storage? PyTorch's Linear layer stores weight as (O, I), so the TritonLinear should do the same.

Now, writing the code.

First, the TritonLinear class:

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize like PyTorch's Linear layer
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _LinearTriton.apply(x, self.weight, self.bias)

Then, the _LinearTriton autograd Function:

class _LinearTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        output = linear_triton_forward(x, weight)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, d_output):
        x, weight = ctx.saved_tensors
        dx, dw = linear_triton_backward(d_output, x, weight)
        db = None
        if ctx.needs_input_grad[2]:
            db = d_output.sum(0)
        return dx, dw, db

Now, the forward and backward Triton functions.

For the forward kernel:

@triton.autotune(...)
@triton.jit
def linear_forward_kernel(...):
    ...

def linear_triton_forward(x, weight):
    # Allocate output tensor
    # Launch kernel with autotuned configs
    # Return output

Similarly for the backward functions.

But implementing the kernels requires careful handling of pointers and block configurations.

For the forward kernel:

The input x is (B, I), weight is (O, I), output is (B, O).

Each element output[b, o] = sum_i x[b, i] * weight[o, i]

In Triton, each program could handle a block of B and O, and accumulate over I in chunks.

But I'm not exactly sure about the indexing. Let's think in terms of 2D blocks.

Maybe split the B dimension into blocks of size BLOCK_SIZE_B, O into BLOCK_SIZE_O, and I into BLOCK_SIZE_I. Then, each block computes a (BLOCK_SIZE_B, BLOCK_SIZE_O) tile of the output, by iterating over BLOCK_SIZE_I chunks of the I dimension.

So, the kernel would have:

- pid_b = program ID along the B dimension
- pid_o = program ID along the O dimension
- For each pid_b and pid_o, compute the tile.

But how to handle the batch dimension? If the batch is large, maybe it's better to have each program handle a portion of the batch and O dimension.

Alternatively, the kernel could use 2D grid: (B // BLOCK_SIZE_B, O // BLOCK_SIZE_O), and each block computes a (BLOCK_SIZE_B, BLOCK_SIZE_O) tile.

Then, for each block, load the corresponding x and weight tiles, compute the dot product, and accumulate into the output.

But the weight is stored as (O, I), so for each o in weight[o, i], we need to load the weight tile for the current O block.

Wait, but the weight's shape is (O, I). So, for a given output column o, the weight row is o.

So, each output element (b, o) is the dot product of x[b, :] and weight[o, :].

So, the kernel can be structured as follows:

- Each program handles a range of b and o indices.
- For each b in the block's B range, and o in the block's O range, accumulate the sum over i of x[b, i] * weight[o, i].

But how to efficiently tile this?

Alternatively, each program could process a group of B rows and O columns, and for each, compute the sum over I. But the weight is (O, I), so each weight row is O elements, but each output element (b, o) uses the o-th row of weight.

Hmm, this seems complex. Maybe using a different approach.

Another way: the matrix multiplication is x (B, I) @ weight.T (I, O) → (B, O). Wait, no: x is (B, I), weight is (O, I), so x @ weight.T would be (B, O), which is correct. But matrix multiplication is (B, I) * (I, O) gives (B, O). Wait, no: (B, I) multiplied by (O, I) transpose (I, O) gives (B, O). Yes.

So, the operation is x @ weight.T. So, the forward pass is a matrix multiplication of x and weight.T.

Therefore, the Triton kernel can be a standard matrix multiplication kernel, where the first matrix is x (B, I), the second matrix is weight.T (I, O), and the output is (B, O).

But Triton's matrix multiplication example can be adapted here. The standard matmul kernel in Triton handles (M, K) @ (K, N) → (M, N). In our case, x is (B, I) (M=B, K=I), weight.T is (I, O) (K=I, N=O). So, the matmul is correct.

Therefore, the forward kernel can be similar to Triton's matmul example, with autotuning for block sizes and group sizes.

But the user specified autotuning for BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. Wait, in matmul terms, BLOCK_SIZE_M (B), BLOCK_SIZE_N (O), and BLOCK_SIZE_K (I).

So, the autotune configs for the forward kernel would explore different combinations of BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, and group sizes.

The same applies to the backward kernels.

For the backward pass, dx is computed as d_output (B, O) @ weight (O, I) → (B, I). So, this is another matmul: (B, O) @ (O, I) → (B, I). For this, the kernel would be a matmul of d_output and weight.

The dw is computed as x.T (I, B) @ d_output (B, O) → (I, O), but since dw in PyTorch is (O, I), we need to transpose it. So, dw = d_output.T @ x (O, B) @ x (B, I) → (O, I). So, this is a matmul of (O, B) and (B, I) → (O, I). So, in Triton, this would require a kernel that multiplies d_output.T (O, B) by x (B, I) → (O, I).

So, the dx kernel is (B, O) @ (O, I) → (B, I), and the dw kernel is (O, B) @ (B, I) → (O, I).

Each of these is a matrix multiplication, so their Triton kernels can be similar to the forward kernel but with different dimensions.

Now, implementing these kernels with autotuning.

For each kernel (forward, dx, dw), we need to define the @triton.autotune decorator with configs that vary the block sizes and group sizes.

For example, for the forward kernel:

configs = [
    triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=4),
    # ... other configs
]

But the actual block size parameters in the kernel would need to match the matmul dimensions.

Wait, perhaps the parameter names should align with the matmul dimensions. For the forward kernel, the matmul is (B, I) @ (I, O) → (B, O), so:

BLOCK_SIZE_M = B (each block handles a block of M dimension, which is B)
BLOCK_SIZE_N = O (block of N dimension, O)
BLOCK_SIZE_K = I (block of K dimension, which is I)

So, the autotune configs would use names like BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K for the respective dimensions.

But the user's instruction mentions BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O. So, perhaps the parameters should be named accordingly, but internally they correspond to M, N, K.

Alternatively, map BLOCK_SIZE_B to BLOCK_SIZE_M, BLOCK_SIZE_O to BLOCK_SIZE_N, and BLOCK_SIZE_I to BLOCK_SIZE_K.

In the kernel code, the parameters are named according to the problem's dimensions for clarity.

So, for the forward kernel, the configs would be:

triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, ...)

Each such config represents the tile sizes for B, O, and I dimensions.

The group size parameters (GROUP_SIZE_*) are used for super blocks, which group multiple blocks together to improve memory access patterns. The group size for each dimension can be autotuned as well.

But how to handle group sizes in Triton? The GROUP_SIZE parameters might control how many blocks are grouped together along each dimension. For example, GROUP_SIZE_B would group multiple BLOCK_SIZE_B blocks along the B dimension.

However, Triton's autotuning might need to explore different group sizes along with block sizes. For example, in the config, group sizes can be added as parameters.

But in the current problem statement, the user mentions using super blocks and autotuning GROUP_SIZE_* parameters. So, each config should include group sizes for the relevant dimensions.

For example, in the forward kernel's config:

triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2}, ...)

This would process a super block of 2x2 blocks in the B and O dimensions.

But integrating this into the kernel requires that the kernel's code processes multiple blocks in a group, perhaps using a loop.

However, implementing super block tiling in Triton might be more complex. Alternatively, the group size could determine the number of blocks processed by a single program in the same group, allowing for better reuse of data in shared memory.

This is getting complicated, and I'm not exactly sure how to structure the super block code. Maybe a better approach is to refer to Triton's documentation or examples on using super-block tiling.

Alternatively, perhaps the user refers to using a group of blocks to compute a larger tile (super block) for better memory locality. For example, processing a group of adjacent blocks in the B and O dimensions together, which allows reusing data loaded into shared memory across multiple blocks.

In Triton, the program's grid is determined by the launch parameters. For super blocks, the grid would be smaller, with each program handling a group of blocks.

But this requires restructuring the kernel to handle multiple output elements per program. For example, instead of each program computing a single block of B x O, it computes a group of blocks.

This can improve efficiency by reusing data from the input matrices across multiple computations.

But implementing this in Triton would require loops within the kernel over the group's blocks.

For example, in the forward kernel:

for b_offset in range(0, GROUP_SIZE_B * BLOCK_SIZE_B, BLOCK_SIZE_B):
    for o_offset in range(0, GROUP_SIZE_O * BLOCK_SIZE_O, BLOCK_SIZE_O):
        compute the tile for (b + b_offset, o + o_offset)

But this is just a rough idea. The actual code would need to manage the indices correctly.

Given the complexity, perhaps the best approach is to model the kernels after Triton's matmul example, which uses tiling and autotuning, and then adapt it to the specific dimensions of the linear layer.

The matmul example in Triton's documentation uses the following approach:

- Each program computes a block of (BLOCK_SIZE_M, BLOCK_SIZE_N) in the output matrix.
- For each such block, the program iterates over the K dimension in blocks of BLOCK_SIZE_K, loading tiles from both input matrices and accumulating the sum.

This approach can be adapted for the forward, dx, and dw kernels.

Now, putting it all together.

For the forward kernel:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64}, num_warps=8),
        # Add more configs here
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    # Compute program IDs for super block
    pid_sb = tl.program_id(axis=0)
    pid_so = tl.program_id(axis=1)
    
    # Super block dimensions
    num_sb = tl.cdiv(B, BLOCK_SIZE_B * GROUP_SIZE_B)
    num_so = tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)
    
    # Iterate over groups
    for gb in range(GROUP_SIZE_B):
        b_start = pid_sb * GROUP_SIZE_B * BLOCK_SIZE_B + gb * BLOCK_SIZE_B
        b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_offsets < B
        
        for go in range(GROUP_SIZE_O):
            o_start = pid_so * GROUP_SIZE_O * BLOCK_SIZE_O + go * BLOCK_SIZE_O
            o_offsets = o_start + tl.arange(0, BLOCK_SIZE_O)
            o_mask = o_offsets < O
            
            # Accumulator
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
            
            # Iterate over I in blocks of BLOCK_SIZE_I
            for i in range(0, I, BLOCK_SIZE_I):
                i_offsets = i + tl.arange(0, BLOCK_SIZE_I)
                i_mask = i_offsets < I
                
                # Load x tile
                x_ptrs = x_ptr + b_offsets[:, None] * stride_xb + i_offsets[None, :] * stride_xi
                x = tl.load(x_ptrs, mask=b_mask[:, None] & i_mask[None, :], other=0.0)
                
                # Load weight tile
                w_ptrs = weight_ptr + o_offsets[:, None] * stride_wo + i_offsets[None, :] * stride_wi
                w = tl.load(w_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
                
                # Compute matrix product
                acc += tl.dot(x, w, acc_dtype=tl.float32)
            
            # Write back accumulator
            output_ptrs = output_ptr + b_offsets[:, None] * stride_ob + o_offsets[None, :] * stride_oo
            tl.store(output_ptrs, acc, mask=b_mask[:, None] & o_mask[None, :])

This is a simplified example and may have errors, but the idea is to have each program handle a super block of B and O dimensions, loop over the group blocks, and accumulate the products over I.

Similarly, for the dx and dw kernels.

Now, the functions linear_triton_forward and linear_triton_backward would launch these kernels with appropriate parameters.

But handling the strides correctly is crucial. For example, the stride_xb is the stride along the B dimension in x (which is I, since x is (B, I)), stride_xi is 1. Similarly for weight, which is (O, I), so stride_wo is I, and stride_wi is 1.

For the output, which is (B, O), strides would be O (stride_ob) and 1 (stride_oo).

The kernel launch parameters would be grid = (num_sb, num_so), where num_sb and num_so are computed as the number of super blocks needed along B and O dimensions.

Now, for the backward kernels:

dx is computed as d_output (B, O) @ weight (O, I) → (B, I). So, this is a matmul of (B, O) and (O, I), resulting in (B, I).

The kernel for dx would be similar to the forward kernel but with different dimensions.

Similarly, dw is computed as d_output.T (O, B) @ x (B, I) → (O, I). This is a matmul of (O, B) and (B, I), resulting in (O, I).

Implementing these requires separate kernels.

For each kernel, the autotune configs will vary BLOCK_SIZE dimensions and GROUP_SIZE.

Putting all this together, the code would involve:

1. The forward Triton kernel with autotuning.
2. The backward Triton kernels for dx and dw with autotuning.
3. Wrapper functions to launch these kernels.
4. The TritonLinear module using the custom autograd Function.

Now, considering potential edge cases, such as when B, I, O are not multiples of the block sizes. The mask in the kernel handles that by checking the boundaries.

Testing is done using the provided code, which uses the benchmark functions to verify correctness and performance.

Finally, the filename should be linear_triton.py, and the code must integrate with the existing benchmark.py and _benchmark_util.py modules.

I need to ensure that the code meets all the requirements, including autotune for block and group sizes in both forward and backward passes, and that the tests can be run as specified.

I should also include the detailed comments as per the user's instruction, explaining each function's purpose, arguments, returns, and explanation steps.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_O': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_O': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_O': 4}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr
):
    """
    Compute matrix multiplication between input and weight matrices using Triton.

    Args:
        x_ptr (tensor): Input tensor of shape (B, I)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        output_ptr (tensor): Output tensor of shape (B, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_*: Stride information for tensor access
        BLOCK_SIZE_* (tl.constexpr): Block sizes for tiling dimensions
        GROUP_SIZE_* (tl.constexpr): Group sizes for super-block tiling

    Explanation:
        1. Calculate super-block indices and ranges
        2. Load input and weight tiles from global memory
        3. Compute partial matrix products using dot product
        4. Accumulate results and store in output tensor
    """
    pid_sb = tl.program_id(0)
    pid_so = tl.program_id(1)
    
    num_sb = tl.cdiv(B, BLOCK_SIZE_B * GROUP_SIZE_B)
    num_so = tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)
    
    for gb in range(GROUP_SIZE_B):
        b_start = pid_sb * GROUP_SIZE_B * BLOCK_SIZE_B + gb * BLOCK_SIZE_B
        b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_offsets < B
        
        for go in range(GROUP_SIZE_O):
            o_start = pid_so * GROUP_SIZE_O * BLOCK_SIZE_O + go * BLOCK_SIZE_O
            o_offsets = o_start + tl.arange(0, BLOCK_SIZE_O)
            o_mask = o_offsets < O
            
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
            
            for i in range(0, I, BLOCK_SIZE_I):
                i_offsets = i + tl.arange(0, BLOCK_SIZE_I)
                i_mask = i_offsets < I
                
                x_ptrs = x_ptr + b_offsets[:, None] * stride_xb + i_offsets[None, :] * stride_xi
                x = tl.load(x_ptrs, mask=b_mask[:, None] & i_mask[None, :], other=0.0)
                
                w_ptrs = weight_ptr + o_offsets[:, None] * stride_wo + i_offsets[None, :] * stride_wi
                w = tl.load(w_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
                
                acc += tl.dot(x, w, acc_dtype=tl.float32)
            
            output_ptrs = output_ptr + b_offsets[:, None] * stride_ob + o_offsets[None, :] * stride_oo
            tl.store(output_ptrs, acc, mask=b_mask[:, None] & o_mask[None, :])

def linear_triton_forward(x, weight):
    B, I = x.shape
    O, _ = weight.shape
    output = torch.empty((B, O), device=x.device, dtype=x.dtype)
    
    grid = lambda opt: (
        triton.cdiv(B, opt['BLOCK_SIZE_B'] * opt['GROUP_SIZE_B']),
        triton.cdiv(O, opt['BLOCK_SIZE_O'] * opt['GROUP_SIZE_O']),
    )
    
    linear_forward_kernel[grid](
        x, weight, output,
        B, I, O,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        GROUP_SIZE_B=1, GROUP_SIZE_O=1
    )
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dx_kernel(
    d_output_ptr, weight_ptr, dx_ptr,
    B, I, O,
    stride_dob, stride_doo,
    stride_wo, stride_wi,
    stride_dxb, stride_dxi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    Compute input gradient using matrix multiplication between d_output and weight.

    Args:
        d_output_ptr (tensor): Gradient tensor of shape (B, O)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        dx_ptr (tensor): Input gradient tensor of shape (B, I)
        B,I,O (int): Tensor dimensions
        stride_*: Stride information for tensor access
        BLOCK_SIZE_* (tl.constexpr): Tiling parameters
        GROUP_SIZE_* (tl.constexpr): Super-block grouping parameters

    Explanation:
        1. Calculate super-block indices and ranges
        2. Load d_output and weight tiles
        3. Compute partial products and accumulate results
        4. Store final gradients in dx tensor
    """
    pid_sb = tl.program_id(0)
    pid_si = tl.program_id(1)
    
    num_sb = tl.cdiv(B, BLOCK_SIZE_B * GROUP_SIZE_B)
    num_si = tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_I)
    
    for gb in range(GROUP_SIZE_B):
        b_start = pid_sb * GROUP_SIZE_B * BLOCK_SIZE_B + gb * BLOCK_SIZE_B
        b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_offsets < B
        
        for gi in range(GROUP_SIZE_I):
            i_start = pid_si * GROUP_SIZE_I * BLOCK_SIZE_I + gi * BLOCK_SIZE_I
            i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
            i_mask = i_offsets < I
            
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
            
            for o in range(0, O, BLOCK_SIZE_O):
                o_offsets = o + tl.arange(0, BLOCK_SIZE_O)
                o_mask = o_offsets < O
                
                dout_ptrs = d_output_ptr + b_offsets[:, None] * stride_dob + o_offsets[None, :] * stride_doo
                dout = tl.load(dout_ptrs, mask=b_mask[:, None] & o_mask[None, :], other=0.0)
                
                w_ptrs = weight_ptr + o_offsets[:, None] * stride_wo + i_offsets[None, :] * stride_wi
                w = tl.load(w_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
                
                acc += tl.dot(dout, w, acc_dtype=tl.float32)
            
            dx_ptrs = dx_ptr + b_offsets[:, None] * stride_dxb + i_offsets[None, :] * stride_dxi
            tl.store(dx_ptrs, acc, mask=b_mask[:, None] & i_mask[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dw_kernel(
    x_ptr, d_output_ptr, dw_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    Compute weight gradient using outer product between x and d_output tensors.

    Args:
        x_ptr (tensor): Input tensor of shape (B, I)
        d_output_ptr (tensor): Gradient tensor of shape (B, O)
        dw_ptr (tensor): Weight gradient tensor of shape (O, I)
        B,I,O (int): Tensor dimensions
        stride_*: Stride information for tensor access
        BLOCK_SIZE_* (tl.constexpr): Tiling parameters
        GROUP_SIZE_* (tl.constexpr): Super-block grouping parameters

    Explanation:
        1. Calculate super-block indices for weight gradients
        2. Load input and gradient tiles
        3. Compute outer product accumulation
        4. Store results in dw tensor
    """
    pid_so = tl.program_id(0)
    pid_si = tl.program_id(1)
    
    num_so = tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)
    num_si = tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_I)
    
    for go in range(GROUP_SIZE_O):
        o_start = pid_so * GROUP_SIZE_O * BLOCK_SIZE_O + go * BLOCK_SIZE_O
        o_offsets = o_start + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_offsets < O
        
        for gi in range(GROUP_SIZE_I):
            i_start = pid_si * GROUP_SIZE_I * BLOCK_SIZE_I + gi * BLOCK_SIZE_I
            i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
            i_mask = i_offsets < I
            
            acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
            
            for b in range(0, B, BLOCK_SIZE_B):
                b_offsets = b + tl.arange(0, BLOCK_SIZE_B)
                b_mask = b_offsets < B
                
                x_ptrs = x_ptr + b_offsets[:, None] * stride_xb + i_offsets[None, :] * stride_xi
                x = tl.load(x_ptrs, mask=b_mask[:, None] & i_mask[None, :], other=0.0)
                
                dout_ptrs = d_output_ptr + b_offsets[:, None] * stride_dob + o_offsets[None, :] * stride_doo
                dout = tl.load(dout_ptrs, mask=b_mask[:, None] & o_mask[None, :], other=0.0)
                
                acc += tl.dot(dout.T, x, acc_dtype=tl.float32)
            
            dw_ptrs = dw_ptr + o_offsets[:, None] * stride_dwo + i_offsets[None, :] * stride_dwi
            tl.store(dw_ptrs, acc, mask=o_mask[:, None] & i_mask[None, :])

def linear_triton_backward(d_output, x, weight):
    B, I = x.shape
    O, _ = weight.shape
    
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    
    grid_dx = lambda opt: (
        triton.cdiv(B, opt['BLOCK_SIZE_B'] * opt['GROUP_SIZE_B']),
        triton.cdiv(I, opt['BLOCK_SIZE_I'] * opt['GROUP_SIZE_I']),
    )
    linear_backward_dx_kernel[grid_dx](
        d_output, weight, dx,
        B, I, O,
        d_output.stride(0), d_output.stride(1),
        weight.stride(0), weight.stride(1),
        dx.stride(0), dx.stride(1)
    )
    
    grid_dw = lambda opt: (
        triton.cdiv(O, opt['BLOCK_SIZE_O'] * opt['GROUP_SIZE_O']),
        triton.cdiv(I, opt['BLOCK_SIZE_I'] * opt['GROUP_SIZE_I']),
    )
    linear_backward_dw_kernel[grid_dw](
        x, d_output, dw,
        B, I, O,
        x.stride(0), x.stride(1),
        d_output.stride(0), d_output.stride(1),
        dw.stride(0), dw.stride(1)
    )
    
    return dx, dw

class _LinearTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight)
        output = linear_triton_forward(x, weight)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        dx, dw = linear_triton_backward(grad_output, x, weight)
        db = grad_output.sum((0,)) if ctx.needs_input_grad[2] else None
        return dx, dw, db

class TritonLinear(torch.nn.Module):
    """
    Triton implementation of a linear layer with autotuned forward/backward passes.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        bias (bool): Include bias term
        device (torch.device): Computation device
        dtype (torch.dtype): Data type for parameters
    
    Explanation:
        1. Initialize weight parameters using Kaiming uniform distribution
        2. Use custom Triton autograd functions for forward/backward
        3. Enable autotuning for different kernel configurations
    """
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _LinearTriton.apply(x, self.weight, self.bias)

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
```bash
python linear_triton.py
```
</test>