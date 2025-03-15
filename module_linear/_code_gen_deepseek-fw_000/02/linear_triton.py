# filename: linear_triton.py

import torch
import triton
import triton.language as tl

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return linear_forward(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input = linear_backward_input(grad_output, weight)
        grad_weight = linear_backward_weight(grad_output, x)
        return grad_input, grad_weight

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    B, I, O,
    x_stride_b, x_stride_i,
    weight_stride_o, weight_stride_i,
    output_stride_b, output_stride_o,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)

    for i in range(0, I, BLOCK_SIZE_I):
        off_i = i + tl.arange(0, BLOCK_SIZE_I)
        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x = tl.load(x_ptr + off_b[:, None] * x_stride_b + off_i[None, :] * x_stride_i, mask=x_mask, other=0.0)
        w_mask = (off_o[:, None] < O) & (off_i[None, :] < I)
        weight = tl.load(weight_ptr + off_o[:, None] * weight_stride_o + off_i[None, :] * weight_stride_i, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, weight, allow_tf32=True)

    output_ptrs = output_ptr + off_b[:, None] * output_stride_b + off_o[None, :] * output_stride_o
    output_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=output_mask)

def linear_forward(x, weight):
    B, I = x.shape
    O, _ = weight.shape
    output = torch.empty((B, O), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
    linear_forward_kernel[grid](
        x, weight, output, B, I, O,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1)
    )
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 16}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_input_kernel(
    grad_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    grad_output_stride_b, grad_output_stride_o,
    weight_stride_o, weight_stride_i,
    grad_input_stride_b, grad_input_stride_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)

    for o in range(0, O, BLOCK_SIZE_B):
        off_o = o + tl.arange(0, BLOCK_SIZE_B)
        go_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
        grad_output = tl.load(grad_output_ptr + off_b[:, None] * grad_output_stride_b + off_o[None, :] * grad_output_stride_o, mask=go_mask, other=0.0)
        w_mask = (off_o[None, :] < O) & (off_i[:, None] < I)
        weight = tl.load(weight_ptr + off_o[None, :] * weight_stride_o + off_i[:, None] * weight_stride_i, mask=w_mask, other=0.0)
        accumulator += tl.dot(grad_output, weight, allow_tf32=True)

    grad_input_ptrs = grad_input_ptr + off_b[:, None] * grad_input_stride_b + off_i[None, :] * grad_input_stride_i
    mask = (off_b[:, None] < B) & (off_i[None, :] < I)
    tl.store(grad_input_ptrs, accumulator.to(tl.float16), mask=mask)

def linear_backward_input(grad_output, weight):
    B, O = grad_output.shape
    I = weight.shape[1]
    grad_input = torch.empty((B, I), device=grad_output.device, dtype=grad_output.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
    linear_backward_input_kernel[grid](
        grad_output, weight, grad_input, B, O, I,
        grad_output.stride(0), grad_output.stride(1),
        weight.stride(0), weight.stride(1),
        grad_input.stride(0), grad_input.stride(1)
    )
    return grad_input

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 128}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_weight_kernel(
    grad_output_ptr, x_ptr, grad_weight_ptr,
    B, I, O,
    grad_output_stride_b, grad_output_stride_o,
    x_stride_b, x_stride_i,
    grad_weight_stride_o, grad_weight_stride_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, GROUP_SIZE_B: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    accumulator = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)

    for b in range(0, B, GROUP_SIZE_B):
        off_b = b + tl.arange(0, GROUP_SIZE_B)
        x_mask = (off_b[:, None] < B) & (off_i[None, :] < I)
        x = tl.load(x_ptr + off_b[:, None] * x_stride_b + off_i[None, :] * x_stride_i, mask=x_mask, other=0.0)
        go_mask = (off_b[:, None] < B) & (off_o[None, :] < O)
        grad_output = tl.load(grad_output_ptr + off_b[:, None] * grad_output_stride_b + off_o[None, :] * grad_output_stride_o, mask=go_mask, other=0.0)
        accumulator += tl.dot(grad_output.T, x, allow_tf32=True)

    grad_weight_ptrs = grad_weight_ptr + off_o[:, None] * grad_weight_stride_o + off_i[None, :] * grad_weight_stride_i
    mask = (off_o[:, None] < O) & (off_i[None, :] < I)
    tl.atomic_add(grad_weight_ptrs, accumulator.to(tl.float16), mask=mask)

def linear_backward_weight(grad_output, x):
    O = grad_output.shape[1]
    I = x.shape[1]
    grad_weight = torch.zeros((O, I), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
    linear_backward_weight_kernel[grid](
        grad_output, x, grad_weight, B=x.shape[0], I=I, O=O,
        grad_output_stride_b=grad_output.stride(0),
        grad_output_stride_o=grad_output.stride(1),
        x_stride_b=x.stride(0),
        x_stride_i=x.stride(1),
        grad_weight_stride_o=grad_weight.stride(0),
        grad_weight_stride_i=grad_weight.stride(1)
    )
    return grad_weight

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
