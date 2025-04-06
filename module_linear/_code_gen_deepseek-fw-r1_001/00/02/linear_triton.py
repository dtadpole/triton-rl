# filename: linear_triton.py

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """..."""  # Keep existing docstring unchanged
    # Existing kernel implementation unchanged


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 256, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def _backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, O, I,
    stride_dob, stride_doo,
    stride_wo, stride_wi,
    stride_dib, stride_dii,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """..."""  # Keep existing docstring unchanged
    # Existing kernel implementation unchanged


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 16, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_M': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _backward_weight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """..."""  # Keep existing docstring unchanged
    # Existing kernel implementation unchanged


class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
        _forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            # Removed explicit GROUP_SIZE_M parameter
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)
        
        # Compute input gradient
        grid_input = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_O']),)
        _backward_input_kernel[grid_input](
            d_output, weight, d_input,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1),
            # Removed explicit GROUP_SIZE_M parameter
        )
        
        # Compute weight gradient
        grid_weight = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),)
        _backward_weight_kernel[grid_weight](
            input, d_output, d_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1),
            # Removed explicit GROUP_SIZE_M parameter
        )
        
        return d_input, d_weight


class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)


if __name__ == "__main__":
    # Existing main block unchanged
