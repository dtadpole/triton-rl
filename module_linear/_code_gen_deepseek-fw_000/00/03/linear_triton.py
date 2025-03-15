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
        5. Compute partial matrix product with tl.dot (transposing weight)
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
        # Fix: Transpose weight matrix for correct matmul dimensions
        acc += tl.dot(input, weight, allow_tf32=True, trans_b=True)
        input_ptrs += GROUP_SIZE_I * stride_input_i
        weight_ptrs += GROUP_SIZE_I * stride_weight_i
        
    acc = acc.to(tl.float16)
    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    tl.store(output_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

# Rest of the file remains unchanged...
