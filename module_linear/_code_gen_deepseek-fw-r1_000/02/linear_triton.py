# filename: linear_triton.py
import torch
import triton
import triton.language as tl
from torch.autograd import Function

class TritonLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O, _ = weight.shape
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        def grid(meta):
            return (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
        
        forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O, _ = weight.shape
        
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)

        # Backward for input gradient
        def grid_dx(meta):
            return (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        
        backward_dx_kernel[grid_dx](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        # Backward for weight gradient
        def grid_dw(meta):
            return (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        
        backward_dw_kernel[grid_dw](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 256}, num_warps=16),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_wo, stride_wi,
    stride_ob, stride_oo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """
    Performs forward pass matrix multiplication for linear layer.

    Args:
        input_ptr (triton.Pointer): Input tensor of shape (B, I).
        weight_ptr (triton.Pointer): Weight tensor of shape (O, I).
        output_ptr (triton.Pointer): Output tensor of shape (B, O) to be written.
        B (int): Batch dimension size.
        I (int): Input feature dimension size.
        O (int): Output feature dimension size.
        stride_ib, stride_ii: Input tensor strides.
        stride_wo, stride_wi: Weight tensor strides.
        stride_ob, stride_oo: Output tensor strides.
        BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I: Block sizes for tiling.

    Explanation:
        1. Compute block indices for B and O dimensions.
        2. Initialize accumulator for output block.
        3. Iterate over I dimension in blocks, loading input and weight tiles.
        4. Compute matrix product and accumulate results.
        5. Store final results to output tensor.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] + i < I)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] + i < I)
        
        input_tile = tl.load(input_ptrs + i * stride_ii, mask=mask_input, other=0.0)
        weight_tile = tl.load(weight_ptrs + i * stride_wi, mask=mask_weight, other=0.0)
        
        accumulator += tl.dot(input_tile, weight_tile.T)
    
    offs_b_out = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o_out = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    mask = (offs_b_out[:, None] < B) & (offs_o_out[None, :] < O)
    tl.store(output_ptr + offs_b_out[:, None] * stride_ob + offs_o_out[None, :] * stride_oo, 
             accumulator.to(tl.float16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_dx_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, I, O,
    stride_go_b, stride_go_o,
    stride_wo, stride_wi,
    stride_gi_b, stride_gi_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
):
    """
    Computes input gradient (dx) via matrix multiplication of grad_output and weight.

    Args:
        grad_output_ptr (triton.Pointer): Gradient output tensor (B, O).
        weight_ptr (triton.Pointer): Weight tensor (O, I).
        grad_input_ptr (triton.Pointer): Gradient input tensor (B, I) to be written.
        B, I, O: Dimension sizes.
        strides: Tensor strides for all pointers.
        BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O: Tiling block sizes.

    Explanation:
        1. Compute block indices for B and I dimensions.
        2. Load grad_output and weight tiles.
        3. Accumulate matrix product over O dimension.
        4. Store result to grad_input.
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    go_ptrs = grad_output_ptr + offs_b[:, None] * stride_go_b + offs_o[None, :] * stride_go_o
    w_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O):
        mask_go = (offs_b[:, None] < B) & (offs_o[None, :] + o < O)
        mask_w = (offs_o[:, None] + o < O) & (offs_i[None, :] < I)
        
        go_tile = tl.load(go_ptrs + o * stride_go_o, mask=mask_go, other=0.0)
        w_tile = tl.load(w_ptrs + o * stride_wo, mask=mask_w, other=0.0)
        
        accumulator += tl.dot(go_tile, w_tile)
    
    gi_ptrs = grad_input_ptr + offs_b[:, None] * stride_gi_b + offs_i[None, :] * stride_gi_i
    mask = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(gi_ptrs, accumulator.to(tl.float16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_dw_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_ib, stride_ii,
    stride_go_b, stride_go_o,
    stride_gw_o, stride_gw_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    """
    Computes weight gradient (dw) via matrix multiplication of input and grad_output.

    Args:
        input_ptr (triton.Pointer): Input tensor (B, I).
        grad_output_ptr (triton.Pointer): Gradient output tensor (B, O).
        grad_weight_ptr (triton.Pointer): Weight gradient tensor (O, I) to be written.
        B, I, O: Dimension sizes.
        strides: Tensor strides for all pointers.
        BLOCK_SIZE_O, BLOCK_SIZE_I, BLOCK_SIZE_B: Tiling block sizes.

    Explanation:
        1. Compute block indices for O and I dimensions.
        2. Load input and grad_output tiles.
        3. Accumulate outer product over B dimension.
        4. Store result to grad_weight.
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    go_ptrs = grad_output_ptr + offs_b[:, None] * stride_go_b + offs_o[None, :] * stride_go_o
    
    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_B):
        mask_input = (offs_b[:, None] + b < B) & (offs_i[None, :] < I)
        mask_go = (offs_b[:, None] + b < B) & (offs_o[None, :] < O)
        
        input_tile = tl.load(input_ptrs + b * stride_ib, mask=mask_input, other=0.0)
        go_tile = tl.load(go_ptrs + b * stride_go_b, mask=mask_go, other=0.0)
        
        accumulator += tl.dot(go_tile.T, input_tile)
    
    gw_ptrs = grad_weight_ptr + offs_o[:, None] * stride_gw_o + offs_i[None, :] * stride_gw_i
    mask = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.store(gw_ptrs, accumulator.to(tl.float16), mask=mask)

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.xavier_normal_(self.weight)
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

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
