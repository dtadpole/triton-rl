# filename: linear_triton.py
import torch
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 2}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_weight_o, stride_weight_i,
    stride_output_b, stride_output_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Compute matrix multiplication between input (B, I) and weight (O, I) to produce output (B, O).

    Args:
        input_ptr (triton.Pointer): Pointer to input tensor of shape (B, I)
        weight_ptr (triton.Pointer): Pointer to weight tensor of shape (O, I)
        output_ptr (triton.Pointer): Pointer to output tensor of shape (B, O)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        strides: Tensor stride information for memory access patterns
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for super-block optimization in batch dimension

    Explanation:
        1. Calculate program IDs for 2D grid (B/O dimensions)
        2. Load input and weight blocks using super-block grouping
        3. Perform matrix multiplication with accumulation
        4. Store results to output tensor
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B * GROUP_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for _ in range(0, I, BLOCK_SIZE_I):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * stride_weight_i
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o
    tl.store(output_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 1}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _backward_dinput_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    stride_grad_output_b, stride_grad_output_o,
    stride_weight_o, stride_weight_i,
    stride_grad_input_b, stride_grad_input_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Compute gradient for input tensor using matrix multiplication between grad_output (B, O) and weight (O, I).

    Args:
        grad_output_ptr (triton.Pointer): Pointer to gradient output tensor of shape (B, O)
        weight_ptr (triton.Pointer): Pointer to weight tensor of shape (O, I)
        grad_input_ptr (triton.Pointer): Pointer to gradient input tensor of shape (B, I)
        B (int): Batch dimension size
        O (int): Output feature dimension size
        I (int): Input feature dimension size
        strides: Tensor stride information for memory access patterns
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for super-block optimization in batch dimension

    Explanation:
        1. Calculate program IDs for 2D grid (B/I dimensions)
        2. Load grad_output and weight blocks using super-block grouping
        3. Perform matrix multiplication with accumulation
        4. Store results to grad_input tensor
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = tl.arange(0, O)
    
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
    
    acc = tl.zeros((BLOCK_SIZE_B * GROUP_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for _ in range(0, O, BLOCK_SIZE_I):
        grad_output_block = tl.load(grad_output_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output_block, weight_block, allow_tf32=True)
        grad_output_ptrs += BLOCK_SIZE_I * stride_grad_output_o
        weight_ptrs += BLOCK_SIZE_I * stride_weight_i
    
    offs_b = pid_b * BLOCK_SIZE_B * GROUP_SIZE_B + tl.arange(0, BLOCK_SIZE_B * GROUP_SIZE_B)
    grad_input_ptrs = grad_input_ptr + offs_b[:, None] * stride_grad_input_b + offs_i[None, :] * stride_grad_input_i
    tl.store(grad_input_ptrs, acc, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 1}, num_warps=8),
    ],
    key=['I', 'O', 'B']
)
@triton.jit
def _backward_dweight_kernel(
    input_ptr, grad_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_grad_output_b, stride_grad_output_o,
    stride_grad_weight_i, stride_grad_weight_o,
    BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, GROUP_SIZE_B: tl.constexpr
):
    """
    Compute gradient for weight tensor using matrix multiplication between input (B, I) and grad_output (B, O).

    Args:
        input_ptr (triton.Pointer): Pointer to input tensor of shape (B, I)
        grad_output_ptr (triton.Pointer): Pointer to gradient output tensor of shape (B, O)
        grad_weight_ptr (triton.Pointer): Pointer to gradient weight tensor of shape (I, O)
        B (int): Batch dimension size
        I (int): Input feature dimension size
        O (int): Output feature dimension size
        strides: Tensor stride information for memory access patterns
        BLOCK_SIZE_I (tl.constexpr): Block size for input feature dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output feature dimension
        GROUP_SIZE_B (tl.constexpr): Group size for super-block optimization in batch dimension

    Explanation:
        1. Calculate program IDs for 2D grid (I/O dimensions)
        2. Load input and grad_output blocks using super-block grouping
        3. Perform outer product accumulation across batch dimension
        4. Atomically add results to grad_weight tensor
    """
    pid_i = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_b = tl.arange(0, B // (GROUP_SIZE_B * BLOCK_SIZE_O)) * (GROUP_SIZE_B * BLOCK_SIZE_O)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    grad_output_ptrs = grad_output_ptr + offs_b[:, None] * stride_grad_output_b + offs_o[None, :] * stride_grad_output_o
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_O), dtype=tl.float32)
    
    for _ in range(0, B, BLOCK_SIZE_O * GROUP_SIZE_B):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I), other=0.0)
        grad_output_block = tl.load(grad_output_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O), other=0.0)
        acc += tl.dot(input_block.T, grad_output_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_input_b
        grad_output_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_grad_output_b
    
    grad_weight_ptr += offs_i[:, None] * stride_grad_weight_i + offs_o[None, :] * stride_grad_weight_o
    tl.atomic_add(grad_weight_ptr, acc, mask=(offs_i[:, None] < I) & (offs_o[None, :] < O))

class _TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = (
            triton.cdiv(B, 256 * 4),  # Using maximum super-block grouping
            triton.cdiv(O, 64),
        )
        
        _forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            GROUP_SIZE_B=4
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
        grid_input = (
            triton.cdiv(B, 256 * 4),
            triton.cdiv(I, 64),
        )
        _backward_dinput_kernel[grid_input](
            grad_output, weight, grad_input,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
            GROUP_SIZE_B=4
        )
        
        # Compute grad_weight
        grid_weight = (
            triton.cdiv(I, 64),
            triton.cdiv(O, 64),
        )
        _backward_dweight_kernel[grid_weight](
            input, grad_output, grad_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
            GROUP_SIZE_B=4
        )
        
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return _TritonLinearFunction.apply(x, self.weight)

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
