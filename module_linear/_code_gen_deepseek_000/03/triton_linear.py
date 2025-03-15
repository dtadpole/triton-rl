# filename: triton_linear.py
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 128}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    """
    Performs forward matrix multiplication of input and weight matrices.

    Args:
        input_ptr (tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tensor): Pointer to output tensor of shape (B, O)
        B, I, O (int): Dimensions of input and output
        stride_*: Memory stride information for tensor access
        BLOCK_SIZE_B, BLOCK_SIZE_O, BLOCK_SIZE_I (tl.constexpr): Block sizes for autotuning

    Explanation:
        1. Compute block indices for batch and output dimensions
        2. Initialize output block accumulator
        3. Loop over input dimension blocks and accumulate matrix products
        4. Store final results to output tensor
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] + i < I)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] + i < I)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * stride_weight_i
    
    offs_b_out = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o_out = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    output_ptrs = output_ptr + offs_b_out[:, None] * stride_output_b + offs_o_out[None, :] * stride_output_o
    mask = (offs_b_out[:, None] < B) & (offs_o_out[None, :] < O)
    tl.store(output_ptrs, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, I, O,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_din_b, stride_din_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    """
    Computes gradient of input for linear layer.

    Args:
        d_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        weight_ptr (tensor): Pointer to weight tensor (O, I)
        d_input_ptr (tensor): Pointer to input gradient tensor (B, I)
        B, I, O (int): Dimension sizes
        stride_*: Memory stride information
        BLOCK_SIZE_B, BLOCK_SIZE_I (tl.constexpr): Block sizes for autotuning

    Explanation:
        1. Calculate block indices for batch and input dimensions
        2. Initialize accumulator
        3. Loop over output dimension blocks and accumulate products
        4. Store final gradient values
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, 128)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, 128):
        dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + (o + offs_o)[None, :] * stride_dout_o
        weight_ptrs = weight_ptr + (o + offs_o)[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
        
        mask_dout = (offs_b[:, None] < B) & ((o + offs_o)[None, :] < O)
        mask_weight = ((o + offs_o)[:, None] < O) & (offs_i[None, :] < I)
        
        dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(dout, weight, allow_tf32=True)
    
    din_ptrs = d_input_ptr + offs_b[:, None] * stride_din_b + offs_i[None, :] * stride_din_i
    mask = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(din_ptrs, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def backward_weight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    """
    Computes gradient of weight matrix for linear layer.

    Args:
        input_ptr (tensor): Pointer to input tensor (B, I)
        d_output_ptr (tensor): Pointer to output gradient tensor (B, O)
        d_weight_ptr (tensor): Pointer to weight gradient tensor (O, I)
        B, I, O (int): Dimension sizes
        stride_*: Memory stride information
        BLOCK_SIZE_O, BLOCK_SIZE_I (tl.constexpr): Block sizes for autotuning

    Explanation:
        1. Initialize block indices for output and input dimensions
        2. Create memory pointers with appropriate offsets
        3. Iterate over batch dimension with atomic accumulation
        4. Store final accumulated gradients
    """
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, 128)
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, 128):
        input_ptrs = input_ptr + (b + offs_b)[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
        dout_ptrs = d_output_ptr + (b + offs_b)[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
        
        mask_input = ((b + offs_b)[:, None] < B) & (offs_i[None, :] < I)
        mask_dout = ((b + offs_b)[:, None] < B) & (offs_o[None, :] < O)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        dout_block = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        
        acc += tl.dot(dout_block.T, input_block, allow_tf32=True)
    
    d_weight_ptrs = d_weight_ptr + offs_o[:, None] * O + offs_i[None, :]
    mask = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.atomic_add(d_weight_ptrs, acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
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
        
        forward_kernel[grid](
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
        grad_input = torch.empty_like(input)
        grad_weight = torch.zeros_like(weight)
        
        # Compute grad_input
        grid_input = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_B']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        backward_input_kernel[grid_input](
            grad_output, weight, grad_input,
            B, I, O,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1)
        )
        
        # Compute grad_weight
        grid_weight = lambda meta: (
            triton.cdiv(O, meta['BLOCK_SIZE_O']),
            triton.cdiv(I, meta['BLOCK_SIZE_I']),
        )
        backward_weight_kernel[grid_weight](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1)
        )
        
        return grad_input, grad_weight

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.kaiming_normal_(self.weight)
        
    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16):
        triton_module = TritonLinear(I, O, device=device, dtype=dtype)
        torch_module = torch.nn.Linear(I, O, bias=False, device=device, dtype=dtype)
        torch_module.weight.data = triton_module.weight.data.clone()
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
