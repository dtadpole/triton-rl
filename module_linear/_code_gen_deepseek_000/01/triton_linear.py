# filename: triton_linear.py
import torch
import triton
import triton.language as tl

class TritonLinear(torch.nn.Module):
    """
    Triton implementation of a Linear layer with autotuned block sizes for forward and backward passes.
    """
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
    def forward(self, x):
        return _TritonLinearFunction.apply(x, self.weight)

class _TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        def grid(meta): return (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
        
        _forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_B=64, BLOCK_SIZE_O=64, BLOCK_SIZE_I=64  # Defaults will be overridden by autotune
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)
        
        # Backward for input gradient
        def grid_in(meta): return (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        _backward_input_kernel[grid_in](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            BLOCK_SIZE_B=64, BLOCK_SIZE_I=64, BLOCK_SIZE_O=64
        )
        
        # Backward for weight gradient
        def grid_w(meta): return (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        _backward_weight_kernel[grid_w](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
            BLOCK_SIZE_O=64, BLOCK_SIZE_I=64, BLOCK_SIZE_B=64
        )
        
        return grad_input, grad_weight, None

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32}, num_warps=4),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def _forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """
    Compute forward pass for linear layer using matrix multiplication.

    Args:
        input_ptr (tensor): Input tensor of shape (B, I)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        output_ptr (tensor): Output tensor of shape (B, O)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        BLOCK_SIZE_B: Block size for batch dimension
        BLOCK_SIZE_O: Block size for output dimension
        BLOCK_SIZE_I: Block size for input feature dimension

    Explanation:
        1. Calculate block indices for batch and output dimensions
        2. Initialize output accumulator
        3. Loop over input feature blocks
        4. Load input and weight blocks with boundary checks
        5. Compute matrix product and accumulate
        6. Store results to output tensor
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    b_start = pid_b * BLOCK_SIZE_B
    o_start = pid_o * BLOCK_SIZE_O
    
    b_offs = b_start + tl.arange(0, BLOCK_SIZE_B)
    o_offs = o_start + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    for i_start in range(0, I, BLOCK_SIZE_I):
        i_offs = i_start + tl.arange(0, BLOCK_SIZE_I)
        mask_b = b_offs < B
        mask_o = o_offs < O
        mask_i = i_offs < I
        
        input_block = tl.load(
            input_ptr + b_offs[:, None] * stride_input_b + i_offs[None, :] * stride_input_i,
            mask=mask_b[:, None] & mask_i[None, :], other=0.0
        )
        
        weight_block = tl.load(
            weight_ptr + o_offs[:, None] * stride_weight_o + i_offs[None, :] * stride_weight_i,
            mask=mask_o[:, None] & mask_i[None, :], other=0.0
        )
        
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
    
    output_ptrs = output_ptr + b_offs[:, None] * stride_output_b + o_offs[None, :] * stride_output_o
    tl.store(output_ptrs, acc.to(input_block.dtype), mask=mask_b[:, None] & mask_o[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32}, num_warps=4),
    ],
    key=['B', 'I', 'O'],
)
@triton.jit
def _backward_input_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, I, O,
    stride_go_b, stride_go_o,
    stride_w_o, stride_w_i,
    stride_gi_b, stride_gi_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
):
    """
    Compute input gradient for linear layer backward pass.

    Args:
        grad_output_ptr (tensor): Gradient of output tensor (B, O)
        weight_ptr (tensor): Weight tensor (O, I)
        grad_input_ptr (tensor): Gradient of input tensor (B, I)
        BLOCK_SIZE_B: Block size for batch dimension
        BLOCK_SIZE_I: Block size for input dimension
        BLOCK_SIZE_O: Block size for output dimension

    Explanation:
        1. Calculate block indices for batch and input dimensions
        2. Initialize accumulator
        3. Loop over output feature blocks
        4. Load gradient output and weight blocks
        5. Compute matrix product and accumulate
        6. Store results to input gradient tensor
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    b_start = pid_b * BLOCK_SIZE_B
    i_start = pid_i * BLOCK_SIZE_I
    
    b_offs = b_start + tl.arange(0, BLOCK_SIZE_B)
    i_offs = i_start + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    for o_start in range(0, O, BLOCK_SIZE_O):
        o_offs = o_start + tl.arange(0, BLOCK_SIZE_O)
        mask_b = b_offs < B
        mask_i = i_offs < I
        mask_o = o_offs < O
        
        grad_output_block = tl.load(
            grad_output_ptr + b_offs[:, None] * stride_go_b + o_offs[None, :] * stride_go_o,
            mask=mask_b[:, None] & mask_o[None, :], other=0.0
        )
        
        weight_block = tl.load(
            weight_ptr + o_offs[:, None] * stride_w_o + i_offs[None, :] * stride_w_i,
            mask=mask_o[:, None] & mask_i[None, :], other=0.0
        )
        
        acc += tl.dot(grad_output_block, weight_block, allow_tf32=True)
    
    grad_input_ptrs = grad_input_ptr + b_offs[:, None] * stride_gi_b + i_offs[None, :] * stride_gi_i
    tl.store(grad_input_ptrs, acc.to(grad_output_block.dtype), mask=mask_b[:, None] & mask_i[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 32, 'BLOCK_SIZE_I': 32, 'BLOCK_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32}, num_warps=4),
    ],
    key=['O', 'I', 'B'],
)
@triton.jit
def _backward_weight_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_in_b, stride_in_i,
    stride_go_b, stride_go_o,
    stride_gw_o, stride_gw_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    """
    Compute weight gradient for linear layer backward pass.

    Args:
        input_ptr (tensor): Input tensor (B, I)
        grad_output_ptr (tensor): Gradient of output tensor (B, O)
        grad_weight_ptr (tensor): Gradient of weight tensor (O, I)
        BLOCK_SIZE_O: Block size for output dimension
        BLOCK_SIZE_I: Block size for input dimension
        BLOCK_SIZE_B: Block size for batch dimension

    Explanation:
        1. Calculate block indices for output and input dimensions
        2. Initialize accumulator
        3. Loop over batch blocks with atomic updates
        4. Load input and gradient output blocks
        5. Compute outer product and accumulate with atomics
        6. Store final results to weight gradient tensor
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    o_start = pid_o * BLOCK_SIZE_O
    i_start = pid_i * BLOCK_SIZE_I
    
    o_offs = o_start + tl.arange(0, BLOCK_SIZE_O)
    i_offs = i_start + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    for b_start in range(0, B, BLOCK_SIZE_B):
        b_offs = b_start + tl.arange(0, BLOCK_SIZE_B)
        mask_b = b_offs < B
        mask_o = o_offs < O
        mask_i = i_offs < I
        
        input_block = tl.load(
            input_ptr + b_offs[:, None] * stride_in_b + i_offs[None, :] * stride_in_i,
            mask=mask_b[:, None] & mask_i[None, :], other=0.0
        )
        
        grad_output_block = tl.load(
            grad_output_ptr + b_offs[:, None] * stride_go_b + o_offs[None, :] * stride_go_o,
            mask=mask_b[:, None] & mask_o[None, :], other=0.0
        )
        
        acc += tl.dot(grad_output_block.T, input_block, allow_tf32=True)
    
    grad_weight_ptrs = grad_weight_ptr + o_offs[:, None] * stride_gw_o + i_offs[None, :] * stride_gw_i
    tl.atomic_add(grad_weight_ptrs, acc.to(input_block.dtype), mask=mask_o[:, None] & mask_i[None, :])

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            torch_module.weight.copy_(triton_module.weight)
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
