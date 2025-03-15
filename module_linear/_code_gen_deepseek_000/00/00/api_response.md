API: deepseek
Test Case: ./module_linear/
Timestamp: 20250314_093815

<think>
Okay, let's see. The user provided an error message when running their TritonLinear code. The task is to fix the code based on the errors without changing the filename or other parts. Let's break down the errors.

First, there are deprecation warnings about using `torch.cuda.amp.custom_fwd` and `custom_bwd`. The messages suggest replacing them with `torch.amp.custom_fwd` and `custom_bwd` with the `device_type='cuda'` argument. So the fix here would be to import `custom_fwd` and `custom_bwd` from `torch.amp` instead of `torch.cuda.amp`, and add the device_type parameter.

Then, there's a NameError: math is not defined. Looking at the code, in the reset_parameters method, they use math.sqrt(5). But the code doesn't import the math module. So adding `import math` at the top should resolve that.

Let me check the original code. The custom_fwd and custom_bwd decorators are applied to the forward and backward methods. The original imports for these decorators were probably from `torch.cuda.amp`, which is now deprecated. Changing the import to `from torch.amp import custom_fwd, custom_bwd` would address the warnings. Also, adding `device_type='cuda'` to the decorators as per the warning's suggestion.

Additionally, the math module wasn't imported, so adding `import math` at the beginning is necessary.

Now, I need to make sure these changes are the only ones made. The filename remains triton_linear.py. The structure of the code shouldn't change otherwise. So the fixes are:

1. Import custom_fwd and custom_bwd from torch.amp instead of torch.cuda.amp.
2. Add device_type='cuda' to both decorators.
3. Import the math module.

These are minimal changes that address the errors without altering other parts of the code.
</think>

```python
# filename: triton_linear.py
import math
import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
from torch.nn.parameter import Parameter

class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @custom_fwd(cast_inputs=torch.float16, device_type='cuda')
    def forward(self, x):
        return _LinearFunction.apply(x, self.weight)

class _LinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16, device_type='cuda')
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return _linear_forward(x, weight)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        return _linear_backward(grad_output, x, weight), None

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1024, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 4096, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128}, num_warps=4),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def _linear_forward_kernel(
    x_ptr, weight_ptr, output_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_wb, stride_wi,
    stride_outputb, stride_outputo,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_o = tl.cdiv(O, BLOCK_SIZE_O)
    pid_b = pid // num_pid_o
    pid_o = pid % num_pid_o

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_i = tl.arange(0, BLOCK_SIZE_I)
    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)

    x_ptrs = x_ptr + off_b[:, None] * stride_xb + off_i[None, :] * stride_xi
    weight_ptrs = weight_ptr + off_o[:, None] * stride_wb + off_i[None, :] * stride_wi

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    for i in range(0, I, BLOCK_SIZE_I):
        x = tl.load(x_ptrs, mask=(off_b[:, None] < B) & (off_i[None, :] < I - i), other=0.0)
        w = tl.load(weight_ptrs, mask=(off_o[:, None] < O) & (off_i[None, :] < I - i), other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_SIZE_I * stride_xi
        weight_ptrs += BLOCK_SIZE_I * stride_wi

    off_output_b = off_b[:, None] * stride_outputb
    off_output_o = off_o[None, :] * stride_outputo
    output_ptrs = output_ptr + off_output_b + off_output_o
    tl.store(output_ptrs, acc.to(tl.float16), mask=(off_b[:, None] < B) & (off_o[None, :] < O))

def _linear_forward(x, weight):
    B, I = x.shape
    O = weight.shape[0]
    output = torch.empty((B, O), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(O, meta['BLOCK_SIZE_O']),)
    _linear_forward_kernel[grid](x, weight, output, B, I, O, 
                               x.stride(0), x.stride(1),
                               weight.stride(0), weight.stride(1),
                               output.stride(0), output.stride(1))
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1024, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 4096, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128}, num_warps=4),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def _linear_backward_kernel_x(
    grad_output_ptr, x_ptr, grad_x_ptr,
    B, O, I,
    stride_grad_outputb, stride_grad_outputo,
    stride_xb, stride_xi,
    stride_grad_xb, stride_grad_xi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
    pid_b = pid // num_pid_i
    pid_i = pid % num_pid_i

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_o = tl.arange(0, BLOCK_SIZE_O)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    grad_output_ptrs = grad_output_ptr + off_b[:, None] * stride_grad_outputb + off_o[None, :] * stride_grad_outputo
    weight_ptrs = x_ptr + off_o[:, None] * stride_xb + off_i[None, :] * stride_xi

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    for o in range(0, O, BLOCK_SIZE_O):
        grad_output = tl.load(grad_output_ptrs, mask=(off_b[:, None] < B) & (off_o[None, :] < O - o), other=0.0)
        weight = tl.load(weight_ptrs, mask=(off_o[:, None] < O) & (off_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output, weight, allow_tf32=True)
        grad_output_ptrs += BLOCK_SIZE_O * stride_grad_outputo
        weight_ptrs += BLOCK_SIZE_O * stride_xb

    off_grad_x_b = off_b[:, None] * stride_grad_xb
    off_grad_x_i = off_i[None, :] * stride_grad_xi
    grad_x_ptrs = grad_x_ptr + off_grad_x_b + off_grad_x_i
    tl.store(grad_x_ptrs, acc.to(tl.float16), mask=(off_b[:, None] < B) & (off_i[None, :] < I))

@triton.jit
def _linear_backward_kernel_weight(
    grad_output_ptr, x_ptr, grad_weight_ptr,
    B, O, I,
    stride_grad_outputb, stride_grad_outputo,
    stride_xb, stride_xi,
    stride_grad_weighto, stride_grad_weighti,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr
):
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    off_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    off_b = tl.arange(0, BLOCK_SIZE_B)

    grad_output_ptrs = grad_output_ptr + off_b[:, None] * stride_grad_outputb + off_o[None, :] * stride_grad_outputo
    x_ptrs = x_ptr + off_b[:, None] * stride_xb + off_i[None, :] * stride_xi

    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    for b in range(0, B, BLOCK_SIZE_B):
        grad_output = tl.load(grad_output_ptrs, mask=(off_b[:, None] < B - b) & (off_o[None, :] < O), other=0.0)
        x = tl.load(x_ptrs, mask=(off_b[:, None] < B - b) & (off_i[None, :] < I), other=0.0)
        acc += tl.dot(grad_output, x, allow_tf32=True)
        grad_output_ptrs += BLOCK_SIZE_B * stride_grad_outputb
        x_ptrs += BLOCK_SIZE_B * stride_xb

    grad_weight_ptrs = grad_weight_ptr + off_o[:, None] * stride_grad_weighto + off_i[None, :] * stride_grad_weighti
    tl.store(grad_weight_ptrs, acc.to(tl.float16), mask=(off_o[:, None] < O) & (off_i[None, :] < I))

def _linear_backward(grad_output, x, weight):
    B, O = grad_output.shape
    I = weight.shape[1]
    grad_x = torch.empty_like(x)
    grad_weight = torch.zeros_like(weight)

    grid_x = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),)
    _linear_backward_kernel_x[grid_x](grad_output, weight, grad_x, B, O, I,
                                    grad_output.stride(0), grad_output.stride(1),
                                    weight.stride(0), weight.stride(1),
                                    grad_x.stride(0), grad_x.stride(1),
                                    BLOCK_SIZE_B=128, BLOCK_SIZE_O=128, BLOCK_SIZE_I=128)

    grid_weight = (triton.cdiv(O, 128), triton.cdiv(I, 128))
    _linear_backward_kernel_weight[grid_weight](grad_output, x, grad_weight, B, O, I,
                                              grad_output.stride(0), grad_output.stride(1),
                                              x.stride(0), x.stride(1),
                                              grad_weight.stride(0), grad_weight.stride(1),
                                              BLOCK_SIZE_B=128, BLOCK_SIZE_O=128, BLOCK_SIZE_I=128)

    return grad_x, grad_weight

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
```