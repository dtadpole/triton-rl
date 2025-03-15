# filename: linear_triton.py

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_warps=8),
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
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr
):
    """
    Computes forward pass for linear layer using block matrix multiplication.

    Args:
        input_ptr (tensor): Input tensor of shape (B, I)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        output_ptr (tensor): Output tensor of shape (B, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_*: Memory strides for corresponding dimensions
        BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O (tl.constexpr): Block sizes for each dimension
        GROUP_SIZE_B (tl.constexpr): Group size for batch dimension super-blocking

    Explanation:
        1. Calculate block indices for batch and output dimensions
        2. Load blocks of input and weight matrices into SRAM
        3. Compute matrix product for current blocks
        4. Accumulate results and write back to output
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_b = pid // num_pid_o
    pid_o = pid % num_pid_o

    offs_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I * GROUP_SIZE_B):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & (offs_i[None, :] < I - i), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I - i), other=0.0)
        accumulator += tl.dot(input_block, weight_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_I * GROUP_SIZE_B * stride_input_i
        weight_ptrs += BLOCK_SIZE_I * GROUP_SIZE_B * stride_weight_i

    offs_output_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_output_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    output_ptrs = output_ptr + offs_output_b[:, None] * stride_output_b + offs_output_o[None, :] * stride_output_o
    output_mask = (offs_output_b[:, None] < B) & (offs_output_o[None, :] < O)
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=output_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def linear_backward_dinput_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, O, I,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_din_b, stride_din_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr
):
    """
    Computes gradient of input for linear layer backward pass.

    Args:
        d_output_ptr (tensor): Gradient of output (B, O)
        weight_ptr (tensor): Weight tensor (O, I)
        d_input_ptr (tensor): Gradient of input (B, I)
        B, O, I (int): Dimensions of tensors
        stride_*: Memory strides for corresponding dimensions
        BLOCK_SIZE_B, BLOCK_SIZE_O (tl.constexpr): Block sizes
        GROUP_SIZE_B (tl.constexpr): Group size for super-blocking

    Explanation:
        1. Calculate block indices for batch and input dimensions
        2. Load blocks of d_output and weight matrices
        3. Compute matrix product for gradient calculation
        4. Accumulate and store results in d_input
    """
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_O)  # Reusing BLOCK_SIZE_O for input dimension
    pid_b = pid // num_pid_i
    pid_i = pid % num_pid_i

    offs_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
    offs_i = pid_i * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O * GROUP_SIZE_B):
        dout_block = tl.load(dout_ptrs, mask=(offs_b[:, None] < B) & (offs_o[None, :] < O - o), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_o[:, None] < O) & (offs_i[None, :] < I), other=0.0)
        accumulator += tl.dot(dout_block, weight_block, allow_tf32=True)
        dout_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_dout_o
        weight_ptrs += BLOCK_SIZE_O * GROUP_SIZE_B * stride_weight_o

    offs_din_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_din_i = pid_i * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    din_ptrs = d_input_ptr + offs_din_b[:, None] * stride_din_b + offs_din_i[None, :] * stride_din_i
    din_mask = (offs_din_b[:, None] < B) & (offs_din_i[None, :] < I)
    tl.store(din_ptrs, accumulator.to(tl.float16), mask=din_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dweight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    stride_dw_o, stride_dw_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr
):
    """
    Computes gradient of weights for linear layer backward pass.

    Args:
        input_ptr (tensor): Input tensor (B, I)
        d_output_ptr (tensor): Gradient of output (B, O)
        d_weight_ptr (tensor): Gradient of weights (O, I)
        B, I, O (int): Dimensions of tensors
        stride_*: Memory strides for corresponding dimensions
        BLOCK_SIZE_B, BLOCK_SIZE_I (tl.constexpr): Block sizes
        GROUP_SIZE_B (tl.constexpr): Group size for super-blocking

    Explanation:
        1. Calculate block indices for output and input dimensions
        2. Load blocks of input and d_output matrices
        3. Compute outer product for gradient accumulation
        4. Atomically update d_weight tensor
    """
    pid = tl.program_id(0)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_B)  # Reuse BLOCK_SIZE_B for O
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    pid_o = pid // num_pid_i
    pid_i = pid % num_pid_i

    offs_o = pid_o * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
    dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_B * GROUP_SIZE_B):
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B - b) & (offs_i[None, :] < I), other=0.0)
        dout_block = tl.load(dout_ptrs, mask=(offs_b[:, None] < B - b) & (offs_o[None, :] < O), other=0.0)
        accumulator += tl.dot(dout_block, input_block, allow_tf32=True)
        input_ptrs += BLOCK_SIZE_B * GROUP_SIZE_B * stride_input_b
        dout_ptrs += BLOCK_SIZE_B * GROUP_SIZE_B * stride_dout_b

    dw_ptrs = d_weight_ptr + offs_o[:, None] * stride_dw_o + offs_i[None, :] * stride_dw_i
    dw_mask = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.atomic_add(dw_ptrs, accumulator.to(tl.float16), mask=dw_mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, I = input.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=input.device, dtype=input.dtype)
        
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
        linear_forward_kernel[grid](
            input, weight, output,
            B, I, O,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight = ctx.saved_tensors
        B, I = input.shape
        O = weight.shape[0]

        d_input = torch.empty_like(input)
        d_weight = torch.zeros_like(weight)

        # Compute d_input
        grid_dinput = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_O']),)
        linear_backward_dinput_kernel[grid_dinput](
            d_output, weight, d_input,
            B, O, I,
            d_output.stride(0), d_output.stride(1),
            weight.stride(0), weight.stride(1),
            d_input.stride(0), d_input.stride(1),
        )

        # Compute d_weight
        grid_dweight = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),)
        linear_backward_dweight_kernel[grid_dweight](
            input, d_output, d_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1),
        )

        return d_input, d_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

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
