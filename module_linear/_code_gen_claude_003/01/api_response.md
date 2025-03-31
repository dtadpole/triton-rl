Timestamp: 20250331_045240

# thinking
Alright, I'm being asked to implement a Triton version of PyTorch's nn.Linear layer. I need to generate both the forward and backward passes and use autotuning for block sizes and super blocks.

Let me first understand what nn.Linear does. The `nn.Linear` module applies a linear transformation to the incoming data: `y = xA^T + b`. If `bias=False`, then the bias term is not included.

For my implementation:
1. I need to create a class called `TritonLinear` that mirrors the functionality of `nn.Linear`.
2. The forward pass should use Triton to compute `y = xA^T`.
3. The backward pass should compute gradients for both input `x` and weight `A`.
4. I should use autotuning to determine the best block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O).
5. I should also use super blocks and autotune the corresponding GROUP_SIZE_* parameters.

Let's make my code as efficient as possible. Here's my structure:

1. Import the necessary libraries
2. Define Triton kernels for forward and backward operations
3. Create a PyTorch module class that uses these Triton kernels
4. Implement forward and backward functions in a way that matches PyTorch's behavior
5. Test the code using the provided testing framework

Let's start with the imports and the Triton kernels:

```python
import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple
```

Now, let's define the forward kernel. This will compute `out = x @ weight.T`. The kernel will need to handle the matrix multiplication efficiently. I'll use a tiled approach with autotuning for the block sizes.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer.
    
    Args:
        x_ptr: Pointer to the input tensor of shape (B, I)
        weight_ptr: Pointer to the weight tensor of shape (O, I)
        output_ptr: Pointer to the output tensor of shape (B, O)
        B: Batch size
        I: Input dimension
        O: Output dimension
        stride_xb: Stride for batch dimension in input
        stride_xi: Stride for input dimension in input
        stride_wo: Stride for output dimension in weight
        stride_wi: Stride for input dimension in weight
        stride_ob: Stride for batch dimension in output
        stride_oo: Stride for output dimension in output
        
    Explanation:
        1. Each program will handle a specific BLOCK_SIZE_B x BLOCK_SIZE_O region of the output.
        2. We load BLOCK_SIZE_B x BLOCK_SIZE_I of x and BLOCK_SIZE_O x BLOCK_SIZE_I of weights.
        3. We perform matrix multiplication between these blocks.
        4. We write the result to the output tensor.
    
    Note:
        This kernel assumes that the weight matrix is transposed for matrix multiplication (out = x @ weight.T).
    """
    # Define super blocks
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    # Number of blocks needed for each dimension
    grid_b = (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_o = (O + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O
    
    # Group size for super blocks
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_bi = GROUP_SIZE_B
    num_pid_oi = GROUP_SIZE_O
    num_pid_bo = tl.cdiv(num_pid_b, num_pid_bi)
    num_pid_oo = tl.cdiv(num_pid_o, num_pid_oi)
    
    # Super block IDs
    pid_b_outer = pid_b // num_pid_bi
    pid_b_inner = pid_b % num_pid_bi
    pid_o_outer = pid_o // num_pid_oi
    pid_o_inner = pid_o % num_pid_oi
    
    # Initialize indices for batch and output dimensions
    # This is the starting point for this program in the output matrix
    start_b = pid_b_outer * (num_pid_bi * BLOCK_SIZE_B) + pid_b_inner * BLOCK_SIZE_B
    start_o = pid_o_outer * (num_pid_oi * BLOCK_SIZE_O) + pid_o_inner * BLOCK_SIZE_O
    
    # Create offset pointers for this program's part of the matrices
    offs_x_batch = start_b + tl.arange(0, BLOCK_SIZE_B)
    offs_w_out = start_o + tl.arange(0, BLOCK_SIZE_O)
    offs_input = tl.arange(0, BLOCK_SIZE_I)
    
    # Create masks to handle edge cases
    mask_b = offs_x_batch < B
    mask_o = offs_w_out < O
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over blocks of the input dimension
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        start_i = i * BLOCK_SIZE_I
        mask_i = (start_i + offs_input) < I
        
        # Load input and weight blocks
        x_block = tl.load(
            x_ptr + offs_x_batch[:, None] * stride_xb + (start_i + offs_input[None, :]) * stride_xi,
            mask=mask_b[:, None] & mask_i[None, :],
            other=0.0
        )
        
        w_block = tl.load(
            weight_ptr + offs_w_out[:, None] * stride_wo + (start_i + offs_input[None, :]) * stride_wi,
            mask=mask_o[:, None] & mask_i[None, :],
            other=0.0
        )
        
        # Compute matrix multiplication for this block
        acc += tl.dot(x_block, w_block.T)
    
    # Write result to output
    output_ptrs = output_ptr + offs_x_batch[:, None] * stride_ob + offs_w_out[None, :] * stride_oo
    tl.store(output_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])
```

Now, let's define the backward kernel for input gradients. This computes `dx = dy @ weight`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
    ],
    key=['B', 'O', 'I'],
)
@triton.jit
def linear_backward_input_kernel(
    # Pointers to matrices
    grad_output_ptr, weight_ptr, grad_input_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_gob, stride_goo,
    stride_wo, stride_wi,
    stride_gib, stride_gii,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    """
    Compute the backward pass for input gradients of a linear layer.
    
    Args:
        grad_output_ptr: Pointer to the output gradient tensor of shape (B, O)
        weight_ptr: Pointer to the weight tensor of shape (O, I)
        grad_input_ptr: Pointer to the input gradient tensor of shape (B, I)
        B: Batch size
        O: Output dimension
        I: Input dimension
        stride_gob: Stride for batch dimension in output gradient
        stride_goo: Stride for output dimension in output gradient
        stride_wo: Stride for output dimension in weight
        stride_wi: Stride for input dimension in weight
        stride_gib: Stride for batch dimension in input gradient
        stride_gii: Stride for input dimension in input gradient
        
    Explanation:
        1. Each program will handle a specific BLOCK_SIZE_B x BLOCK_SIZE_I region of the input gradient.
        2. We load BLOCK_SIZE_B x BLOCK_SIZE_O of grad_output and BLOCK_SIZE_O x BLOCK_SIZE_I of weights.
        3. We perform matrix multiplication: grad_input = grad_output @ weight.
        4. We write the result to the grad_input tensor.
    """
    # Define super blocks
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Number of blocks needed for each dimension
    grid_b = (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    grid_i = (I + BLOCK_SIZE_I - 1) // BLOCK_SIZE_I
    
    # Group size for super blocks
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_bi = GROUP_SIZE_B
    num_pid_ii = GROUP_SIZE_I
    num_pid_bo = tl.cdiv(num_pid_b, num_pid_bi)
    num_pid_io = tl.cdiv(num_pid_i, num_pid_ii)
    
    # Super block IDs
    pid_b_outer = pid_b // num_pid_bi
    pid_b_inner = pid_b % num_pid_bi
    pid_i_outer = pid_i // num_pid_ii
    pid_i_inner = pid_i % num_pid_ii
    
    # Initialize indices for batch and input dimensions
    # This is the starting point for this program in the grad_input matrix
    start_b = pid_b_outer * (num_pid_bi * BLOCK_SIZE_B) + pid_b_inner * BLOCK_SIZE_B
    start_i = pid_i_outer * (num_pid_ii * BLOCK_SIZE_I) + pid_i_inner * BLOCK_SIZE_I
    
    # Create offset pointers for this program's part of the matrices
    offs_go_batch = start_b + tl.arange(0, BLOCK_SIZE_B)
    offs_gi_in = start_i + tl.arange(0, BLOCK_SIZE_I)
    offs_out = tl.arange(0, BLOCK_SIZE_O)
    
    # Create masks to handle edge cases
    mask_b = offs_go_batch < B
    mask_i = offs_gi_in < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over blocks of the output dimension
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        start_o = o * BLOCK_SIZE_O
        mask_o = (start_o + offs_out) < O
        
        # Load grad_output and weight blocks
        grad_output_block = tl.load(
            grad_output_ptr + offs_go_batch[:, None] * stride_gob + (start_o + offs_out[None, :]) * stride_goo,
            mask=mask_b[:, None] & mask_o[None, :],
            other=0.0
        )
        
        weight_block = tl.load(
            weight_ptr + (start_o + offs_out[:, None]) * stride_wo + offs_gi_in[None, :] * stride_wi,
            mask=mask_o[:, None] & mask_i[None, :],
            other=0.0
        )
        
        # Compute matrix multiplication for this block: grad_input = grad_output @ weight
        acc += tl.dot(grad_output_block, weight_block)
    
    # Write result to grad_input
    grad_input_ptrs = grad_input_ptr + offs_go_batch[:, None] * stride_gib + offs_gi_in[None, :] * stride_gii
    tl.store(grad_input_ptrs, acc, mask=mask_b[:, None] & mask_i[None, :])
```

Now, let's define the backward kernel for weight gradients. This computes `dw = dy.T @ x`.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['O', 'I', 'B'],
)
@triton.jit
def linear_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr, input_ptr, grad_weight_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_gob, stride_goo,
    stride_ib, stride_ii,
    stride_gwo, stride_gwi,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for weight gradients of a linear layer.
    
    Args:
        grad_output_ptr: Pointer to the output gradient tensor of shape (B, O)
        input_ptr: Pointer to the input tensor of shape (B, I)
        grad_weight_ptr: Pointer to the weight gradient tensor of shape (O, I)
        B: Batch size
        O: Output dimension
        I: Input dimension
        stride_gob: Stride for batch dimension in output gradient
        stride_goo: Stride for output dimension in output gradient
        stride_ib: Stride for batch dimension in input
        stride_ii: Stride for input dimension in input
        stride_gwo: Stride for output dimension in weight gradient
        stride_gwi: Stride for input dimension in weight gradient
        
    Explanation:
        1. Each program will handle a specific BLOCK_SIZE_O x BLOCK_SIZE_I region of the weight gradient.
        2. We load BLOCK_SIZE_B blocks of grad_output (B, O) and input (B, I).
        3. We perform matrix multiplication: grad_weight = grad_output.T @ input.
        4. We accumulate the result across batches.
        5. We write the result to the grad_weight tensor.
    """
    # Define super blocks
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Number of blocks needed for each dimension
    grid_o = (O + BLOCK_SIZE_O - 1) // BLOCK_SIZE_O
    grid_i = (I + BLOCK_SIZE_I - 1) // BLOCK_SIZE_I
    
    # Group size for super blocks
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_oi = GROUP_SIZE_O
    num_pid_ii = GROUP_SIZE_I
    num_pid_oo = tl.cdiv(num_pid_o, num_pid_oi)
    num_pid_io = tl.cdiv(num_pid_i, num_pid_ii)
    
    # Super block IDs
    pid_o_outer = pid_o // num_pid_oi
    pid_o_inner = pid_o % num_pid_oi
    pid_i_outer = pid_i // num_pid_ii
    pid_i_inner = pid_i % num_pid_ii
    
    # Initialize indices for output and input dimensions
    # This is the starting point for this program in the grad_weight matrix
    start_o = pid_o_outer * (num_pid_oi * BLOCK_SIZE_O) + pid_o_inner * BLOCK_SIZE_O
    start_i = pid_i_outer * (num_pid_ii * BLOCK_SIZE_I) + pid_i_inner * BLOCK_SIZE_I
    
    # Create offset pointers for this program's part of the matrices
    offs_go_out = start_o + tl.arange(0, BLOCK_SIZE_O)
    offs_in = start_i + tl.arange(0, BLOCK_SIZE_I)
    offs_batch = tl.arange(0, BLOCK_SIZE_B)
    
    # Create masks to handle edge cases
    mask_o = offs_go_out < O
    mask_i = offs_in < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over blocks of the batch dimension
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        start_b = b * BLOCK_SIZE_B
        mask_b = (start_b + offs_batch) < B
        
        # Load grad_output and input blocks
        grad_output_block = tl.load(
            grad_output_ptr + (start_b + offs_batch)[None, :] * stride_gob + offs_go_out[:, None] * stride_goo,
            mask=mask_o[:, None] & mask_b[None, :],
            other=0.0
        )
        
        input_block = tl.load(
            input_ptr + (start_b + offs_batch)[None, :] * stride_ib + offs_in[:, None] * stride_ii,
            mask=mask_i[:, None] & mask_b[None, :],
            other=0.0
        )
        
        # Compute outer product: grad_weight += grad_output.T @ input
        # We need to transpose grad_output in the dot operation
        acc += tl.dot(grad_output_block, input_block.T)
    
    # Write result to grad_weight
    grad_weight_ptrs = grad_weight_ptr + offs_go_out[:, None] * stride_gwo + offs_in[None, :] * stride_gwi
    tl.store(grad_weight_ptrs, acc, mask=mask_o[:, None] & mask_i[None, :])
```

Now, let's define a PyTorch module that uses these Triton kernels.

```python
class TritonLinear(torch.nn.Module):
    """
    A linear layer implemented using Triton.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None):
        """
        Initialize a TritonLinear layer.
        
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            bias (bool, optional): If set to True, the layer will learn an additive bias. Default: False
            device (torch.device, optional): Device to place the parameters. Default: None
            dtype (torch.dtype, optional): Data type of the parameters. Default: None
        
        Explanation:
            1. Initialize the parent class (torch.nn.Module).
            2. Store the input parameters (in_features, out_features, bias).
            3. Create a parameter tensor for the weights with shape (out_features, in_features).
            4. Create a parameter tensor for the bias with shape (out_features,) if bias=True.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TritonLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We initialize weight and bias the same way as PyTorch does
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters the same way as PyTorch
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Reset the parameters (weight and bias) using Kaiming uniform initialization.
        
        Explanation:
            1. Initialize the weight using Kaiming uniform initialization with a='math.sqrt(5)'.
            2. Initialize the bias, if it exists, by uniform sampling within bounds determined by fan_in.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def get_parameter(self, name: str) -> torch.nn.Parameter:
        """
        Get a parameter by name.
        
        Args:
            name (str): The name of the parameter ('weight' or 'bias').
        
        Returns:
            torch.nn.Parameter: The requested parameter.
        """
        if name == 'weight':
            return self.weight
        elif name == 'bias':
            return self.bias
        else:
            raise ValueError(f"Parameter {name} not found")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        
        Explanation:
            1. Get the dimensions of the input tensor.
            2. Allocate memory for the output tensor.
            3. Call the Triton kernel to compute the matrix multiplication: output = input @ weight.T.
            4. If bias exists, add it to the output.
        """
        B, I = input.shape
        O = self.out_features
        
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        # Call the forward kernel
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
        )
        
        linear_forward_kernel[grid](
            input, self.weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
        )
        
        # Add bias if needed
        if self.bias is not None:
            output += self.bias
        
        return output
    
    class LinearBackward(torch.autograd.Function):
        """
        Autograd function for the TritonLinear layer.
        """
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            """
            Forward pass for the autograd function.
            
            Args:
                ctx: Context for saving variables for the backward pass
                input (torch.Tensor): Input tensor of shape (batch_size, in_features)
                weight (torch.Tensor): Weight tensor of shape (out_features, in_features)
                bias (torch.Tensor, optional): Bias tensor of shape (out_features,)
            
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_features)
                
            Explanation:
                1. Save input and weight for the backward pass.
                2. Get the dimensions of the input tensor.
                3. Allocate memory for the output tensor.
                4. Call the Triton kernel to compute the matrix multiplication: output = input @ weight.T.
                5. If bias exists, add it to the output.
            """
            ctx.save_for_backward(input, weight, bias)
            
            B, I = input.shape
            O = weight.shape[0]
            
            output = torch.empty((B, O), device=input.device, dtype=input.dtype)
            
            # Call the forward kernel
            grid = lambda meta: (
                triton.cdiv(B, meta['BLOCK_SIZE_B']),
                triton.cdiv(O, meta['BLOCK_SIZE_O']),
            )
            
            linear_forward_kernel[grid](
                input, weight, output,
                B, I, O,
                input.stride(0), input.stride(1),
                weight.stride(0), weight.stride(1),
                output.stride(0), output.stride(1),
            )
            
            if bias is not None:
                output += bias
            
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass for the autograd function.
            
            Args:
                ctx: Context containing saved variables from the forward pass
                grad_output (torch.Tensor): Gradient of the loss with respect to the output
            
            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Gradients with respect to input, weight, and bias
                
            Explanation:
                1. Retrieve saved tensors from context.
                2. Compute gradient with respect to input: grad_input = grad_output @ weight.
                3. Compute gradient with respect to weight: grad_weight = grad_output.T @ input.
                4. Compute gradient with respect to bias (if exists): grad_bias = grad_output.sum(0).
            """
            input, weight, bias = ctx.saved_tensors
            
            B, I = input.shape
            O = weight.shape[0]
            
            # Allocate memory for gradients
            grad_input = torch.empty_like(input)
            grad_weight = torch.empty_like(weight)
            
            # Compute gradient with respect to input: grad_input = grad_output @ weight
            grid_input = lambda meta: (
                triton.cdiv(B, meta['BLOCK_SIZE_B']),
                triton.cdiv(I, meta['BLOCK_SIZE_I']),
            )
            
            linear_backward_input_kernel[grid_input](
                grad_output, weight, grad_input,
                B, O, I,
                grad_output.stride(0), grad_output.stride(1),
                weight.stride(0), weight.stride(1),
                grad_input.stride(0), grad_input.stride(1),
            )
            
            # Compute gradient with respect to weight: grad_weight = grad_output.T @ input
            grid_weight = lambda meta: (
                triton.cdiv(O, meta['BLOCK_SIZE_O']),
                triton.cdiv(I, meta['BLOCK_SIZE_I']),
            )
            
            linear_backward_weight_kernel[grid_weight](
                grad_output, input, grad_weight,
                B, O, I,
                grad_output.stride(0), grad_output.stride(1),
                input.stride(0), input.stride(1),
                grad_weight.stride(0), grad_weight.stride(1),
            )
            
            # Compute gradient with respect to bias
            grad_bias = None
            if bias is not None:
                grad_bias = grad_output.sum(0)
            
            return grad_input, grad_weight, grad_bias
```

Let's add a method to enable autograd for this layer.

```python
def forward(self, input: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the linear layer with autograd support.
    
    Args:
        input (torch.Tensor): Input tensor of shape (batch_size, in_features)
    
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_features)
    """
    return self.LinearBackward.apply(input, self.weight, self.bias)
```

Now, let's put everything together and make sure the code is complete and ready for testing.

I noticed that I need to update the forward method to use the autograd function. Let me do that and combine everything into a single module. I'll also add the main function that uses the benchmark utilities to test correctness and performance as specified.

Let me review what I have:

1. I have a triton kernel for the forward pass.
2. I have a triton kernel for computing the gradient with respect to input.
3. I have a triton kernel for computing the gradient with respect to weights.
4. I have a PyTorch module that uses these kernels.
5. I have an autograd function to handle backpropagation.

Now I need to combine these into a single linear_triton.py file and make sure everything works correctly.

I notice some redundancy in my code and I can simplify it. Specifically, the forward method in the TritonLinear class can just call the autograd function's forward method. Let me clean that up.

Let me review my entire implementation to make sure it's correct and ready for testing:

1. The kernel names and arguments are correct.
2. The grid functions for launching kernels are correct.
3. The LinearBackward autograd function properly saves and uses the context.
4. The TritonLinear class correctly initializes and uses the parameters.
5. The main function correctly uses the benchmark utilities.

I need to check a few things in the main function:
- It correctly generates modules and inputs for testing.
- It calls the correct benchmark functions.
- It tests both forward and backward passes.

I believe the code is correct, but let me make one more check. I'm particularly concerned about the backward pass, since that's often trickier to get right.

In the backward pass, we need to compute:
1. `grad_input = grad_output @ weight`
2. `grad_weight = grad_output.T @ input`
3. If bias exists, `grad_bias = grad_output.sum(0)`

My kernels and autograd function look correct for these operations.

I'm also concerned about making sure I'm using the kernels correctly. For the forward pass, the grid should span the batch size and output features. For the backward pass, the grid for computing grad_input should span batch size and input features, and the grid for computing grad_weight should span output features and input features.

Additionally, I need to make sure I'm handling the strides correctly in the kernels, since incorrect stride calculations can lead to subtle and hard-to-debug errors.

I believe my implementation addresses all these concerns. Let's proceed with the final implementation.

# text
<think>
# Triton Linear Layer Implementation Analysis

I need to implement a Triton-based version of PyTorch's `nn.Linear` layer, with both forward and backward pass functionality. This implementation will leverage Triton's parallel computing capabilities and autotuning to maximize performance.

## Components Required

1. **Forward Kernel**: Matrix multiplication between input tensor and transposed weight matrix
2. **Backward Input Kernel**: Compute gradients for inputs (dX) using output gradients and weights
3. **Backward Weight Kernel**: Compute gradients for weights (dW) using output gradients and inputs
4. **TritonLinear Module**: PyTorch module wrapper that uses these Triton kernels
5. **Autograd Function**: Custom autograd implementation for backward computation

## Technical Considerations

- **Autotuning**: I'll implement autotuning for block sizes in all kernels (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O)
- **Super Blocks**: Will use super blocks with GROUP_SIZE parameters to better utilize GPU resources
- **GEMM Optimization**: Will implement efficient matrix multiplication with proper tiling
- **Stride Handling**: Need to carefully manage tensor strides for correct memory access patterns
- **Edge Case Handling**: Implement proper masking for handling tensor dimensions that aren't multiples of block sizes

## Implementation Strategy

For each kernel:
1. Define a grid of autotuned configurations
2. Use proper tiling for matrix multiplications
3. Handle edge cases with masks
4. Manage shared memory to maximize throughput

The module will follow PyTorch's parameter initialization conventions and provide a seamless drop-in replacement for nn.Linear.

Let me proceed with the implementation.
</think>

```python
# filename: linear_triton.py

import torch
import triton
import triton.language as tl
import math
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_O': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    B, I, O,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_xb, stride_xi,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_O: tl.constexpr,
):
    """
    Compute the forward pass of a linear layer with matrix multiplication: out = x @ weight.T

    Args:
        x_ptr (ptr): Pointer to the input tensor of shape (B, I)
        weight_ptr (ptr): Pointer to the weight tensor of shape (O, I)
        output_ptr (ptr): Pointer to the output tensor of shape (B, O)
        B (int): Batch size
        I (int): Input dimension
        O (int): Output dimension
        stride_xb (int): Stride for batch dimension in input
        stride_xi (int): Stride for input dimension in input
        stride_wo (int): Stride for output dimension in weight
        stride_wi (int): Stride for input dimension in weight
        stride_ob (int): Stride for batch dimension in output
        stride_oo (int): Stride for output dimension in output

    Returns:
        None: Results written to output_ptr

    Explanation:
        1. Calculate program and block indices for the super block structure
        2. Create offset pointers for this program's part of the matrices with proper masking
        3. Initialize accumulator for the dot product results
        4. Iterate through blocks of the input dimension, loading input and weight data
        5. Compute matrix multiplication using tiled approach and accumulate results
        6. Write final accumulated results to the output tensor
    """
    # Define super block structure
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    # Super block calculations for improved parallelism
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_bi = GROUP_SIZE_B
    num_pid_oi = GROUP_SIZE_O
    num_pid_bo = tl.cdiv(num_pid_b, num_pid_bi)
    num_pid_oo = tl.cdiv(num_pid_o, num_pid_oi)
    
    # Super block IDs
    pid_b_outer = pid_b // num_pid_bi
    pid_b_inner = pid_b % num_pid_bi
    pid_o_outer = pid_o // num_pid_oi
    pid_o_inner = pid_o % num_pid_oi
    
    # Initialize indices for batch and output dimensions
    start_b = pid_b_outer * (num_pid_bi * BLOCK_SIZE_B) + pid_b_inner * BLOCK_SIZE_B
    start_o = pid_o_outer * (num_pid_oi * BLOCK_SIZE_O) + pid_o_inner * BLOCK_SIZE_O
    
    # Create offset pointers for this program's part of the matrices
    offs_x_batch = start_b + tl.arange(0, BLOCK_SIZE_B)
    offs_w_out = start_o + tl.arange(0, BLOCK_SIZE_O)
    offs_input = tl.arange(0, BLOCK_SIZE_I)
    
    # Create masks to handle edge cases
    mask_b = offs_x_batch < B
    mask_o = offs_w_out < O
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    # Iterate over blocks of the input dimension
    for i in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
        start_i = i * BLOCK_SIZE_I
        mask_i = (start_i + offs_input) < I
        
        # Load input and weight blocks with masking for edge cases
        x_block = tl.load(
            x_ptr + offs_x_batch[:, None] * stride_xb + (start_i + offs_input[None, :]) * stride_xi,
            mask=mask_b[:, None] & mask_i[None, :],
            other=0.0
        )
        
        w_block = tl.load(
            weight_ptr + offs_w_out[:, None] * stride_wo + (start_i + offs_input[None, :]) * stride_wi,
            mask=mask_o[:, None] & mask_i[None, :],
            other=0.0
        )
        
        # Compute matrix multiplication for this block
        acc += tl.dot(x_block, w_block.T)
    
    # Write result to output
    output_ptrs = output_ptr + offs_x_batch[:, None] * stride_ob + offs_w_out[None, :] * stride_oo
    tl.store(output_ptrs, acc, mask=mask_b[:, None] & mask_o[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8}, num_warps=8),
    ],
    key=['B', 'O', 'I'],
)
@triton.jit
def linear_backward_input_kernel(
    # Pointers to matrices
    grad_output_ptr, weight_ptr, grad_input_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_gob, stride_goo,
    stride_wo, stride_wi,
    stride_gib, stride_gii,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr,
):
    """
    Compute the backward pass for input gradients of a linear layer: grad_input = grad_output @ weight

    Args:
        grad_output_ptr (ptr): Pointer to the output gradient tensor of shape (B, O)
        weight_ptr (ptr): Pointer to the weight tensor of shape (O, I)
        grad_input_ptr (ptr): Pointer to the input gradient tensor of shape (B, I)
        B (int): Batch size
        O (int): Output dimension
        I (int): Input dimension
        stride_gob (int): Stride for batch dimension in output gradient
        stride_goo (int): Stride for output dimension in output gradient
        stride_wo (int): Stride for output dimension in weight
        stride_wi (int): Stride for input dimension in weight
        stride_gib (int): Stride for batch dimension in input gradient
        stride_gii (int): Stride for input dimension in input gradient

    Returns:
        None: Results written to grad_input_ptr

    Explanation:
        1. Calculate program and block indices for the super block structure
        2. Create offset pointers for this program's part of the matrices with proper masking
        3. Initialize accumulator for the gradient calculation
        4. Iterate through blocks of the output dimension
        5. Load gradient outputs and weights with masking for edge cases
        6. Compute matrix multiplication to determine input gradients
        7. Write final computed gradients to the input gradient tensor
    """
    # Define super block structure
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Super block calculations for improved parallelism
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_bi = GROUP_SIZE_B
    num_pid_ii = GROUP_SIZE_I
    num_pid_bo = tl.cdiv(num_pid_b, num_pid_bi)
    num_pid_io = tl.cdiv(num_pid_i, num_pid_ii)
    
    # Super block IDs
    pid_b_outer = pid_b // num_pid_bi
    pid_b_inner = pid_b % num_pid_bi
    pid_i_outer = pid_i // num_pid_ii
    pid_i_inner = pid_i % num_pid_ii
    
    # Initialize indices for batch and input dimensions
    start_b = pid_b_outer * (num_pid_bi * BLOCK_SIZE_B) + pid_b_inner * BLOCK_SIZE_B
    start_i = pid_i_outer * (num_pid_ii * BLOCK_SIZE_I) + pid_i_inner * BLOCK_SIZE_I
    
    # Create offset pointers for this program's part of the matrices
    offs_go_batch = start_b + tl.arange(0, BLOCK_SIZE_B)
    offs_gi_in = start_i + tl.arange(0, BLOCK_SIZE_I)
    offs_out = tl.arange(0, BLOCK_SIZE_O)
    
    # Create masks to handle edge cases
    mask_b = offs_go_batch < B
    mask_i = offs_gi_in < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over blocks of the output dimension
    for o in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        start_o = o * BLOCK_SIZE_O
        mask_o = (start_o + offs_out) < O
        
        # Load grad_output and weight blocks
        grad_output_block = tl.load(
            grad_output_ptr + offs_go_batch[:, None] * stride_gob + (start_o + offs_out[None, :]) * stride_goo,
            mask=mask_b[:, None] & mask_o[None, :],
            other=0.0
        )
        
        weight_block = tl.load(
            weight_ptr + (start_o + offs_out[:, None]) * stride_wo + offs_gi_in[None, :] * stride_wi,
            mask=mask_o[:, None] & mask_i[None, :],
            other=0.0
        )
        
        # Compute matrix multiplication for input gradients: grad_input = grad_output @ weight
        acc += tl.dot(grad_output_block, weight_block)
    
    # Write result to grad_input
    grad_input_ptrs = grad_input_ptr + offs_go_batch[:, None] * stride_gib + offs_gi_in[None, :] * stride_gii
    tl.store(grad_input_ptrs, acc, mask=mask_b[:, None] & mask_i[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 8, 'GROUP_SIZE_I': 8, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['O', 'I', 'B'],
)
@triton.jit
def linear_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr, input_ptr, grad_weight_ptr,
    # Matrix dimensions
    B, O, I,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_gob, stride_goo,
    stride_ib, stride_ii,
    stride_gwo, stride_gwi,
    # Meta-parameters
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr,
):
    """
    Compute the backward pass for weight gradients of a linear layer: grad_weight = grad_output.T @ input

    Args:
        grad_output_ptr (ptr): Pointer to the output gradient tensor of shape (B, O)
        input_ptr (ptr): Pointer to the input tensor of shape (B, I)
        grad_weight_ptr (ptr): Pointer to the weight gradient tensor of shape (O, I)
        B (int): Batch size
        O (int): Output dimension
        I (int): Input dimension
        stride_gob (int): Stride for batch dimension in output gradient
        stride_goo (int): Stride for output dimension in output gradient
        stride_ib (int): Stride for batch dimension in input
        stride_ii (int): Stride for input dimension in input
        stride_gwo (int): Stride for output dimension in weight gradient
        stride_gwi (int): Stride for input dimension in weight gradient

    Returns:
        None: Results written to grad_weight_ptr

    Explanation:
        1. Calculate program and block indices for the super block structure
        2. Create offset pointers for this program's part of the matrices with proper masking
        3. Initialize accumulator for the weight gradient calculation
        4. Iterate through blocks of the batch dimension
        5. Load gradient outputs and inputs with masking for edge cases
        6. Compute outer product to determine weight gradients and accumulate over the batch
        7. Write accumulated weight gradients to the weight gradient tensor
    """
    # Define super block structure
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Super block calculations for improved parallelism
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_oi = GROUP_SIZE_O
    num_pid_ii = GROUP_SIZE_I
    num_pid_oo = tl.cdiv(num_pid_o, num_pid_oi)
    num_pid_io = tl.cdiv(num_pid_i, num_pid_ii)
    
    # Super block IDs
    pid_o_outer = pid_o // num_pid_oi
    pid_o_inner = pid_o % num_pid_oi
    pid_i_outer = pid_i // num_pid_ii
    pid_i_inner = pid_i % num_pid_ii
    
    # Initialize indices for output and input dimensions
    start_o = pid_o_outer * (num_pid_oi * BLOCK_SIZE_O) + pid_o_inner * BLOCK_SIZE_O
    start_i = pid_i_outer * (num_pid_ii * BLOCK_SIZE_I) + pid_i_inner * BLOCK_SIZE_I
    
    # Create offset pointers for this program's part of the matrices
    offs_go_out = start_o + tl.arange(0, BLOCK_SIZE_O)
    offs_in = start_i + tl.arange(0, BLOCK_SIZE_I)
    offs_batch = tl.arange(0, BLOCK_SIZE_B)
    
    # Create masks to handle edge cases
    mask_o = offs_go_out < O
    mask_i = offs_in < I
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    # Iterate over blocks of the batch dimension
    for b in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        start_b = b * BLOCK_SIZE_B
        mask_b = (start_b + offs_batch) < B
        
        # Load grad_output and input blocks
        grad_output_block = tl.load(
            grad_output_ptr + (start_b + offs_batch)[None, :] * stride_gob + offs_go_out[:, None] * stride_goo,
            mask=mask_o[:, None] & mask_b[None, :],
            other=0.0
        )
        
        input_block = tl.load(
            input_ptr + (start_b + offs_batch)[None, :] * stride_ib + offs_in[:, None] * stride_ii,
            mask=mask_i[:, None] & mask_b[None, :],
            other=0.0
        )
        
        # Compute outer product for weight gradients: grad_weight += grad_output.T @ input
        acc += tl.dot(grad_output_block, input_block.T)
    
    # Write result to grad_weight
    grad_weight_ptrs = grad_weight_ptr + offs_go_out[:, None] * stride_gwo + offs_in[None, :] * stride_gwi
    tl.store(grad_weight_ptrs, acc, mask=mask_o[:, None] & mask_i[None, :])

class TritonLinear(torch.nn.Module):
    """
    A linear layer implemented using Triton kernels.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None):
        """
        Initialize a TritonLinear layer.
        
        Args:
            in_features (int): Size of each input sample
            out_features (int): Size of each output sample
            bias (bool): If set to True, the layer will learn an additive bias. Default: False
            device (torch.device): Device to place the parameters
            dtype (torch.dtype): Data type of the parameters
            
        Returns:
            None
            
        Explanation:
            1. Initialize the parent class (torch.nn.Module)
            2. Store the input parameters (in_features, out_features)
            3. Create a parameter tensor for the weights with shape (out_features, in_features)
            4. Create a parameter tensor for the bias with shape (out_features,) if bias=True
            5. Initialize parameters using Kaiming uniform initialization
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TritonLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weight parameter
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters the same way as PyTorch
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """
        Reset the parameters (weight and bias) using Kaiming uniform initialization.
        
        Args:
            None
            
        Returns:
            None
            
        Explanation:
            1. Initialize the weight using Kaiming uniform initialization with a=sqrt(5)
            2. If bias exists, initialize it by uniform sampling within bounds determined by fan_in
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def get_parameter(self, name: str) -> torch.nn.Parameter:
        """
        Get a parameter by name.
        
        Args:
            name (str): The name of the parameter ('weight' or 'bias')
        
        Returns:
            torch.nn.Parameter: The requested parameter
            
        Explanation:
            1. Check if the requested parameter name is valid
            2. Return the corresponding parameter if it exists
        """
        if name == 'weight':
            return self.weight
        elif name == 'bias':
            return self.bias
        else:
            raise ValueError(f"Parameter {name} not found")
    
    class LinearFunction(torch.autograd.Function):
        """
        Autograd function for the TritonLinear layer.
        """
        @staticmethod
        def forward(ctx, input, weight, bias=None):
            """
            Forward pass for the autograd function.
            
            Args:
                ctx (Context): Context object for storing tensors needed in backward
                input (torch.Tensor): Input tensor of shape (B, I) [B×I]
                weight (torch.Tensor): Weight tensor of shape (O, I) [O×I]
                bias (torch.Tensor, optional): Bias tensor of shape (O,) [O]
            
            Returns:
                torch.Tensor: Output tensor of shape (B, O) [B×O]
                
            Explanation:
                1. Save input and weight for the backward pass
                2. Get the dimensions of the input tensor
                3. Allocate memory for the output tensor
                4. Launch the Triton kernel to compute matrix multiplication
                5. If bias exists, add it to the output
                6. Return the computed output
            """
            ctx.save_for_backward(input, weight, bias)
            
            B, I = input.shape
            O = weight.shape[0]
            
            output = torch.empty((B, O), device=input.device, dtype=input.dtype)
            
            # Launch the forward kernel
            grid = lambda meta: (
                triton.cdiv(B, meta['BLOCK_SIZE_B']),
                triton.cdiv(O, meta['BLOCK_SIZE_O']),
            )
            
            linear_forward_kernel[grid](
                input, weight, output,
                B, I, O,
                input.stride(0), input.stride(1),
                weight.stride(0), weight.stride(1),
                output.stride(0), output.stride(1),
            )
            
            if bias is not None:
                output += bias
            
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass for the autograd function.
            
            Args:
                ctx (Context): Context object containing saved tensors from forward
                grad_output (torch.Tensor): Gradient of the loss with respect to the output (B, O) [B×O]
            
            Returns:
                Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Gradients with respect to input, weight, and bias
                
            Explanation:
                1. Retrieve saved tensors from context
                2. Get the dimensions of the tensors
                3. Allocate memory for gradients
                4. Launch Triton kernel to compute gradient with respect to input
                5. Launch Triton kernel to compute gradient with respect to weight
                6. Compute gradient with respect to bias if it exists
                7. Return all gradients
            """
            input, weight, bias = ctx.saved_tensors
            
            B, I = input.shape
            O = weight.shape[0]
            
            # Allocate memory for gradients
            grad_input = torch.empty_like(input)
            grad_weight = torch.empty_like(weight)
            
            # Compute gradient with respect to input: grad_input = grad_output @ weight
            grid_input = lambda meta: (
                triton.cdiv(B, meta['BLOCK_SIZE_B']),
                triton.cdiv(I, meta['BLOCK_SIZE_I']),
            )
            
            linear_backward_input_kernel[grid_input](
                grad_output, weight, grad_input,
                B, O, I,
                grad_output.stride(0), grad_output.stride(1),
                weight.stride(0), weight.stride(1),
                grad_input.stride(0), grad_input.stride(1),
            )
            
            # Compute gradient with respect to weight: grad_weight = grad_output.T @ input
            grid_weight = lambda meta: (
                triton.cdiv(O, meta['BLOCK_SIZE_O']),
                triton.cdiv(I, meta['BLOCK_SIZE_I']),
            )
            
            linear_backward_weight_kernel[grid_weight](
                grad_output, input, grad_weight,
                B, O, I,
                grad_output.stride(0), grad_output.stride(1),
                input.stride(0), input.stride(1),
                grad_weight.stride(0), grad_weight.stride(1),
            )
            
            # Compute gradient with respect to bias
            grad_bias = None
            if bias is not None:
                grad_bias = grad_output.sum(0)
            
            return grad_input, grad_weight, grad_bias
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer with autograd support.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features) [B×I]
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features) [B×O]
            
        Explanation:
            1. Call the custom autograd function to compute the forward pass
            2. This ensures proper gradient computation during backpropagation
            3. The LinearFunction handles both the forward computation and gradient calculation
        """
        return self.LinearFunction.apply(input, self.weight, self.bias)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, bias=False, device=device, dtype=dtype)
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
# Testing Instructions for TritonLinear Implementation

To test the TritonLinear implementation, follow these steps:

1. Make sure you have PyTorch and Triton installed on your system with CUDA support.
2. The benchmark.py file should already exist in your working directory (this file contains utilities for verifying correctness and benchmarking performance).
3. Run the linear_triton.py script to test both forward and backward pass correctness and performance.

The tests will:
- Verify that TritonLinear produces results that match PyTorch's nn.Linear implementation
- Benchmark performance of both implementations across various sizes
- Generate visualization of the performance differences
- Save results to JSON files for future reference

Run the following command:

```bash
python linear_triton.py
```

The output will show correctness test results (✓ passed or ✗ failed) for each matrix size configuration, followed by performance benchmarks showing execution times and speedup ratios.

The code will generate several output files:
- code.correctness.forward.json - correctness results for forward pass
- code.correctness.backward.json - correctness results for backward pass
- code.performance.forward.json - performance results for forward pass
- code.performance.backward.json - performance results for backward pass
- code.performance.forward.png - performance visualization for forward pass
- code.performance.backward.png - performance visualization for backward pass
</test>

