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
