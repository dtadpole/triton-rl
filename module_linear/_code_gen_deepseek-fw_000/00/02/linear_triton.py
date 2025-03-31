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
    """Compute matrix multiplication between input and weight matrices using Triton."""
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
    )
    return output

# ... (rest of the code remains unchanged)
