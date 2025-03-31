# text

Respond in the following format, start with a "think" section, followed by one code block in markdown, and followed by a "test" section.
    In the "think" section, give comprehensive, logically structured thinking, use markdown formatting for clear information organization, admit uncertainties for ambiguous situations.
    In the code block, generate one python file. The first line of the code block is always a comment starts with "# filename: " and follows with the filename.
        Assume benchmark.py already exists (do NOT generate benchmark.py), and has corresponding verify_correctness_* and benchmark_performance_* functions;
        Generate main code, and call the functions in benchmark.py.
    In the test section, generate instructions and test command to test the code, where the command in a bash code block, do NOT generate pip install command.
<think>
...
</think>
```python
# filename: file_1.py
...
```
```python
# filename: file_2.py
...
```
```python
# filename: file_x.py
...
```
<test>
...
```bash
...
```
</test>


# text

For each Python and Triton function, generate a detailed comment section structured as follows:

Description: A summary of the function's purpose.
Args: A list of arguments, each with a type hint and description.  If a type is a tensor or triton pointer, show the shape.
Returns: The return type(s) and description, including None if applicable.
Explanation: A numbered list of sequential steps the function performs.

Follow this template:
```python
def function_name(arg1: type, arg2: type) -> return_type:
    """
    [Description of the function's purpose.]

    Args:
        arg1 (type): [Description of arg1.] [Shape if tensor or triton pointer]
        arg2 (type): [Description of arg2.] [Shape if tensor or triton pointer]

    Returns:
        return_type: [Description of returned value(s).] [Shape if tensor or triton pointer]

    Explanation:
        1. [Step 1: What the function does first.]
        2. [Step 2: Next action or logic.]
        3. [Additional steps as needed...]
    """
    # Function code here
```


# text
Generate Triton module (TritonLinear) for PyTorch nn.linear, implement both forward and backward. (filename: linear_triton.py)
    use autotune for block sizes (BLOCK_SIZE_B, BLOCK_SIZE_I, BLOCK_SIZE_O) in both forward and backward kernels, use super blocks and autotune corresponding GROUP_SIZE_*,
    test correctness and performance using following code literally:
```python
if __name__ == "__main__":
    B_sizes = [256, 1024, 4096]
    I_sizes = [256, 1024, 4096]
    O_sizes = [256, 1024, 4096]

    def module_generator(B, I, O, device='cuda', dtype=torch.float16)
        triton_module = TritonLinear(I, O, device='cuda', dtype=torch.float16)
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


# file
```python
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
    """
    Compute matrix multiplication between input and weight matrices using Triton.

    Args:
        x_ptr (tensor): Input tensor of shape (B, I)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        output_ptr (tensor): Output tensor of shape (B, O)
        B (int): Batch size dimension
        I (int): Input feature dimension
        O (int): Output feature dimension
        stride_*: Stride information for tensor access
        BLOCK_SIZE_* (tl.constexpr): Block sizes for tiling dimensions
        GROUP_SIZE_* (tl.constexpr): Group sizes for super-block tiling

    Explanation:
        1. Calculate super-block indices and ranges
        2. Load input and weight tiles from global memory
        3. Compute partial matrix products using dot product
        4. Accumulate results and store in output tensor
    """
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
        GROUP_SIZE_B=1, GROUP_SIZE_O=1
    )
    return output

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32, 'GROUP_SIZE_B': 1, 'GROUP_SIZE_I': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 64, 'GROUP_SIZE_B': 2, 'GROUP_SIZE_I': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 128, 'GROUP_SIZE_B': 4, 'GROUP_SIZE_I': 4}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dx_kernel(
    d_output_ptr, weight_ptr, dx_ptr,
    B, I, O,
    stride_dob, stride_doo,
    stride_wo, stride_wi,
    stride_dxb, stride_dxi,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    Compute input gradient using matrix multiplication between d_output and weight.

    Args:
        d_output_ptr (tensor): Gradient tensor of shape (B, O)
        weight_ptr (tensor): Weight tensor of shape (O, I)
        dx_ptr (tensor): Input gradient tensor of shape (B, I)
        B,I,O (int): Tensor dimensions
        stride_*: Stride information for tensor access
        BLOCK_SIZE_* (tl.constexpr): Tiling parameters
        GROUP_SIZE_* (tl.constexpr): Super-block grouping parameters

    Explanation:
        1. Calculate super-block indices and ranges
        2. Load d_output and weight tiles
        3. Compute partial products and accumulate results
        4. Store final gradients in dx tensor
    """
    pid_sb = tl.program_id(0)
    pid_si = tl.program_id(1)
    
    num_sb = tl.cdiv(B, BLOCK_SIZE_B * GROUP_SIZE_B)
    num_si = tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_I)
    
    for gb in range(GROUP_SIZE_B):
        b_start = pid_sb * GROUP_SIZE_B * BLOCK_SIZE_B + gb * BLOCK_SIZE_B
        b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
        b_mask = b_offsets < B
        
        for gi in range(GROUP_SIZE_I):
            i_start = pid_si * GROUP_SIZE_I * BLOCK_SIZE_I + gi * BLOCK_SIZE_I
            i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
            i_mask = i_offsets < I
            
            acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
            
            for o in range(0, O, BLOCK_SIZE_O):
                o_offsets = o + tl.arange(0, BLOCK_SIZE_O)
                o_mask = o_offsets < O
                
                dout_ptrs = d_output_ptr + b_offsets[:, None] * stride_dob + o_offsets[None, :] * stride_doo
                dout = tl.load(dout_ptrs, mask=b_mask[:, None] & o_mask[None, :], other=0.0)
                
                w_ptrs = weight_ptr + o_offsets[:, None] * stride_wo + i_offsets[None, :] * stride_wi
                w = tl.load(w_ptrs, mask=o_mask[:, None] & i_mask[None, :], other=0.0)
                
                acc += tl.dot(dout, w, acc_dtype=tl.float32)
            
            dx_ptrs = dx_ptr + b_offsets[:, None] * stride_dxb + i_offsets[None, :] * stride_dxi
            tl.store(dx_ptrs, acc, mask=b_mask[:, None] & i_mask[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32, 'GROUP_SIZE_O': 1, 'GROUP_SIZE_I': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_B': 64, 'GROUP_SIZE_O': 2, 'GROUP_SIZE_I': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 128, 'GROUP_SIZE_O': 4, 'GROUP_SIZE_I': 4}, num_warps=8),
    ],
    key=['B', 'I', 'O']
)
@triton.jit
def linear_backward_dw_kernel(
    x_ptr, d_output_ptr, dw_ptr,
    B, I, O,
    stride_xb, stride_xi,
    stride_dob, stride_doo,
    stride_dwo, stride_dwi,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_O: tl.constexpr, GROUP_SIZE_I: tl.constexpr
):
    """
    Compute weight gradient using outer product between x and d_output tensors.

    Args:
        x_ptr (tensor): Input tensor of shape (B, I)
        d_output_ptr (tensor): Gradient tensor of shape (B, O)
        dw_ptr (tensor): Weight gradient tensor of shape (O, I)
        B,I,O (int): Tensor dimensions
        stride_*: Stride information for tensor access
        BLOCK_SIZE_* (tl.constexpr): Tiling parameters
        GROUP_SIZE_* (tl.constexpr): Super-block grouping parameters

    Explanation:
        1. Calculate super-block indices for weight gradients
        2. Load input and gradient tiles
        3. Compute outer product accumulation
        4. Store results in dw tensor
    """
    pid_so = tl.program_id(0)
    pid_si = tl.program_id(1)
    
    num_so = tl.cdiv(O, BLOCK_SIZE_O * GROUP_SIZE_O)
    num_si = tl.cdiv(I, BLOCK_SIZE_I * GROUP_SIZE_I)
    
    for go in range(GROUP_SIZE_O):
        o_start = pid_so * GROUP_SIZE_O * BLOCK_SIZE_O + go * BLOCK_SIZE_O
        o_offsets = o_start + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_offsets < O
        
        for gi in range(GROUP_SIZE_I):
            i_start = pid_si * GROUP_SIZE_I * BLOCK_SIZE_I + gi * BLOCK_SIZE_I
            i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
            i_mask = i_offsets < I
            
            acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
            
            for b in range(0, B, BLOCK_SIZE_B):
                b_offsets = b + tl.arange(0, BLOCK_SIZE_B)
                b_mask = b_offsets < B
                
                x_ptrs = x_ptr + b_offsets[:, None] * stride_xb + i_offsets[None, :] * stride_xi
                x = tl.load(x_ptrs, mask=b_mask[:, None] & i_mask[None, :], other=0.0)
                
                dout_ptrs = d_output_ptr + b_offsets[:, None] * stride_dob + o_offsets[None, :] * stride_doo
                dout = tl.load(dout_ptrs, mask=b_mask[:, None] & o_mask[None, :], other=0.0)
                
                acc += tl.dot(dout.T, x, acc_dtype=tl.float32)
            
            dw_ptrs = dw_ptr + o_offsets[:, None] * stride_dwo + i_offsets[None, :] * stride_dwi
            tl.store(dw_ptrs, acc, mask=o_mask[:, None] & i_mask[None, :])

def linear_triton_backward(d_output, x, weight):
    B, I = x.shape
    O, _ = weight.shape
    
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    
    grid_dx = lambda opt: (
        triton.cdiv(B, opt['BLOCK_SIZE_B'] * opt['GROUP_SIZE_B']),
        triton.cdiv(I, opt['BLOCK_SIZE_I'] * opt['GROUP_SIZE_I']),
    )
    linear_backward_dx_kernel[grid_dx](
        d_output, weight, dx,
        B, I, O,
        d_output.stride(0), d_output.stride(1),
        weight.stride(0), weight.stride(1),
        dx.stride(0), dx.stride(1)
    )
    
    grid_dw = lambda opt: (
        triton.cdiv(O, opt['BLOCK_SIZE_O'] * opt['GROUP_SIZE_O']),
        triton.cdiv(I, opt['BLOCK_SIZE_I'] * opt['GROUP_SIZE_I']),
    )
    linear_backward_dw_kernel[grid_dw](
        x, d_output, dw,
        B, I, O,
        x.stride(0), x.stride(1),
        d_output.stride(0), d_output.stride(1),
        dw.stride(0), dw.stride(1)
    )
    
    return dx, dw

class _LinearTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight)
        output = linear_triton_forward(x, weight)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        dx, dw = linear_triton_backward(grad_output, x, weight)
        db = grad_output.sum((0,)) if ctx.needs_input_grad[2] else None
        return dx, dw, db

class TritonLinear(torch.nn.Module):
    """
    Triton implementation of a linear layer with autotuned forward/backward passes.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        bias (bool): Include bias term
        device (torch.device): Computation device
        dtype (torch.dtype): Data type for parameters
    
    Explanation:
        1. Initialize weight parameters using Kaiming uniform distribution
        2. Use custom Triton autograd functions for forward/backward
        3. Enable autotuning for different kernel configurations
    """
    def __init__(self, in_features, out_features, bias=False, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _LinearTriton.apply(x, self.weight, self.bias)

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
```)


# file
```python
# filename: benchmark.py

import torch
import triton
import triton.testing
import numpy as np
import json
import matplotlib.pyplot as plt
from _benchmark_util import (
    _generate_configs,
    _check_outputs_match,
    _copy_parameters,
    _backward_pass,
    _benchmark_implementations,
    _plot_benchmark_results
)


def verify_correctness_func(func_triton, func_torch, input_generator, dimensions, device='cuda', dtype=torch.float16, atol=1e-2, rtol=1e-2, seed=42):
    """
    Verify the function correctness of triton implementation against torch implementation
    """
    torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    correctness = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "device": device_name,
        "results": []
    }

    for config in _generate_configs(dimensions):
        inputs = input_generator(*config, device=device, dtype=dtype)
        triton_output = func_triton(*inputs) if isinstance(inputs, tuple) else func_triton(inputs)
        torch_output = func_torch(*inputs) if isinstance(inputs, tuple) else func_torch(inputs)
        _check_outputs_match(config, correctness, triton_output, torch_output, dtype, atol=atol, rtol=rtol, label="func")

    # write correctness results to file
    with open("code.correctness.func.json", "w") as f:
        f.write(json.dumps(correctness, indent=4))

    return correctness

def verify_correctness_forward(module_generator, input_generator, dimensions, device='cuda', dtype=torch.float16, atol=1e-2, rtol=1e-2, seed=42):
    """
    Verify the forward pass
    """
    torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    correctness = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "device": device_name,
        "results": []
    }

    for config in _generate_configs(dimensions):
        triton_module, torch_module = module_generator(*config, device=device, dtype=dtype)
        _copy_parameters(torch_module, triton_module)

        inputs = input_generator(*config, device=device, dtype=dtype)
        triton_output = triton_module(*inputs) if isinstance(inputs, tuple) else triton_module(inputs)
        torch_output = torch_module(*inputs) if isinstance(inputs, tuple) else torch_module(inputs)
        _check_outputs_match(config, correctness, triton_output, torch_output, dtype, atol=atol, rtol=rtol, label="forward")

    # write correctness results to file
    with open("code.correctness.forward.json", "w") as f:
        f.write(json.dumps(correctness, indent=4))

    return correctness


def verify_correctness_backward(module_generator, backward_generator, dimensions, device='cuda', dtype=torch.float16, atol=1e-2, rtol=1e-2, seed=42):
    """
    Verify the backward pass
    """
    torch.manual_seed(seed)
    device_name = torch.cuda.get_device_name(0)

    correctness = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "device": device_name,
        "results": []
    }

    for config in _generate_configs(dimensions):
        # input generator returns params (for Module initialization) and inputs (for Module forward)
        triton_module, torch_module = module_generator(*config, device=device, dtype=dtype)
        _copy_parameters(torch_module, triton_module)

        inputs, dy = backward_generator(*config, device=device, dtype=dtype)

        dx_triton, dw_triton = _backward_pass(triton_module, inputs, dy)
        dx_torch, dw_torch = _backward_pass(torch_module, inputs, dy)

        # Compare gradients
        _check_outputs_match(config, correctness, dx_triton, dx_torch, dtype, atol=atol, rtol=rtol, label="backward dx")
        _check_outputs_match(config, correctness, dw_triton, dw_torch, dtype, atol=atol, rtol=rtol, label="backward dw")

    # write correctness results to file
    with open("code.correctness.backward.json", "w") as f:
        f.write(json.dumps(correctness, indent=4))

    return correctness


def benchmark_performance_func(func_triton, func_torch, input_generator, dimensions, num_warmup=25, num_repeats=100, device='cuda', dtype=torch.float16):
    """
    Benchmarks the performance of the Triton implementation against PyTorch.
    """
    torch.manual_seed(0)
    configs = _generate_configs(dimensions)

    timings_triton = []
    timings_pytorch = []
    speedups = []
    device_name = torch.cuda.get_device_name(0)

    performance = {
        "speedup": 0.0,
        "device": device_name,
        "results": []
    }

    for config in configs:
        inputs = input_generator(*config, device=device, dtype=dtype)

        _benchmark_implementations(
            config,
            lambda inputs: func_triton(*inputs) if isinstance(inputs, tuple) else func_triton(inputs),
            lambda inputs: func_torch(*inputs) if isinstance(inputs, tuple) else func_torch(inputs),
            inputs,
            timings_triton,
            timings_pytorch,
            speedups,
            performance,
            num_warmup=num_warmup,
            num_repeats=num_repeats
        )

    performance["speedup"] = np.median(speedups)

    # write performance results to file
    with open("code.performance.func.json", "w") as f:
        f.write(json.dumps(performance, indent=4))

    # Generate performance comparison graph
    _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label="func")

    return np.median(timings_triton), np.median(timings_pytorch), np.median(speedups), performance

def benchmark_performance_forward(module_generator, forward_generator, dimensions, num_warmup=25, num_repeats=100, device='cuda', dtype=torch.float16):
    """
    Benchmarks the performance of forward pass
    """
    configs = _generate_configs(dimensions)

    timings_triton = []
    timings_pytorch = []
    speedups = []
    device_name = torch.cuda.get_device_name(0)

    performance = {
        "speedup": 0.0,
        "device": device_name,
        "results": []
    }

    for config in configs:
        triton_module, torch_module = module_generator(*config, device=device, dtype=dtype)
        triton_compiled = torch.compile(triton_module, mode="max-autotune")
        pytorch_compiled = torch.compile(torch_module, mode="max-autotune")

        inputs = forward_generator(*config, device=device, dtype=dtype)

        _benchmark_implementations(
            config,
            lambda inputs: triton_compiled(*inputs) if isinstance(inputs, tuple) else triton_compiled(inputs),
            lambda inputs: pytorch_compiled(*inputs) if isinstance(inputs, tuple) else pytorch_compiled(inputs),
            inputs,
            timings_triton,
            timings_pytorch,
            speedups,
            performance,
            num_warmup=num_warmup,
            num_repeats=num_repeats
        )

    performance["speedup"] = np.median(speedups)

    # write performance results to file
    with open("code.performance.forward.json", "w") as f:
        f.write(json.dumps(performance, indent=4))

    # Generate performance comparison graph
    _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label="forward")

    return np.median(timings_triton), np.median(timings_pytorch), np.median(speedups), performance


def benchmark_performance_backward(module_generator, input_generator, dimensions, num_warmup=25, num_repeats=100, device='cuda', dtype=torch.float16):
    """
    Benchmarks the performance of backward pass
    """
    configs = _generate_configs(dimensions)

    timings_triton = []
    timings_pytorch = []
    speedups = []
    device_name = torch.cuda.get_device_name(0)

    performance = {
        "speedup": 0.0,
        "device": device_name,
        "results": []
    }

    for config in configs:
        triton_module, torch_module = module_generator(*config, device=device, dtype=dtype)
        triton_compiled = torch.compile(triton_module, mode="max-autotune")
        pytorch_compiled = torch.compile(torch_module, mode="max-autotune")

        inputs, dy = input_generator(*config, device=device, dtype=dtype)

        _benchmark_implementations(
            config,
            lambda x: _backward_pass(triton_compiled, x[0], x[1]),
            lambda x: _backward_pass(pytorch_compiled, x[0], x[1]),
            (inputs, dy),
            timings_triton,
            timings_pytorch,
            speedups,
            performance,
            num_warmup=num_warmup,
            num_repeats=num_repeats
        )

    performance["speedup"] = np.median(speedups)

    # write performance results to file
    with open("code.performance.backward.json", "w") as f:
        f.write(json.dumps(performance, indent=4))

    # Generate performance comparison graph
    _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label="backward")

    return np.median(timings_triton), np.median(timings_pytorch), np.median(speedups), performance
```)


# file
```python
import torch
import triton
import numpy as np
import json
import matplotlib.pyplot as plt

def _generate_configs(dims):
    """
    Generate all possible combinations of dimensions for benchmarking.

    Args:
        dims (List[List[int]]): List of lists containing possible values for each dimension.
            Example: [[16, 32], [128, 256], [512, 1024]] represents batch_sizes, input_dims, output_dims.

    Returns:
        List[List[int]]: List of all possible dimension combinations.
            Example: [[16, 128, 512], [16, 128, 1024], ..., [32, 256, 1024]]

    Example:
        >>> dims = [[16, 32], [128, 256], [512, 1024]]
        >>> _generate_configs(dims)
        [[16, 128, 512], [16, 128, 1024], [16, 256, 512], [16, 256, 1024],
         [32, 128, 512], [32, 128, 1024], [32, 256, 512], [32, 256, 1024]]
    """
    configs = []
    indices = np.zeros(len(dims), dtype=np.int32)

    # increment indices
    def increment_indices():
        for i in range(len(dims) - 1, -1, -1):
            if indices[i] < len(dims[i]) - 1:
                indices[i] += 1
                return True
            indices[i] = 0
        return False

    while True:
        stack = []
        for i, dim in enumerate(dims):
            stack.append(dim[indices[i]])
        configs.append(stack.copy())
        if not increment_indices():
            break

    return configs

def _check_outputs_match(config, correctness, triton_output, torch_output, dtype, atol=1e-2, rtol=1e-2, label=""):
    """
    Check if outputs from Triton and PyTorch implementations match within tolerance.

    Args:
        config (List[int]): Current configuration being tested (e.g., [batch_size, input_dim, output_dim]).
        correctness (Dict): Dictionary to track test results with keys:
            - passed (int): Number of passed tests
            - failed (int): Number of failed tests
            - total (int): Total number of tests
            - results (List): Detailed results for each test
        triton_output (torch.Tensor | Tuple[torch.Tensor]): Output from Triton implementation.
        torch_output (torch.Tensor | Tuple[torch.Tensor]): Output from PyTorch implementation.
        dtype (torch.dtype): Data type used for tensors.
        atol (float, optional): Absolute tolerance. Defaults to 1e-2.
        rtol (float, optional): Relative tolerance. Defaults to 1e-2.
        label (str, optional): Label for the test (e.g., "forward", "backward"). Defaults to "".

    Returns:
        bool: True if outputs match within tolerance, False otherwise.
    """
    mismatch = False
    if isinstance(torch_output, tuple):
        for i in range(len(torch_output)):
            if torch.allclose(triton_output[i], torch_output[i], atol=atol, rtol=rtol):
                pass
            else:
                mismatch = True
                break
    else:
        if torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
            pass
        else:
            mismatch = True

    if not mismatch:
        print(f"✅ Correctness test passed for shape ({config}, {dtype}) [{label}]")
        correctness["passed"] += 1
        correctness["total"] += 1
        correctness["results"].append({
            "shape": config,
            "dtype": str(dtype),
            "correctness": True
        })
    else:
        print(f"❌ Correctness test failed for shape ({config}, {dtype}) [{label}]")
        correctness["failed"] += 1
        correctness["total"] += 1
        correctness["results"].append({
            "shape": config,
            "dtype": str(dtype),
            "correctness": False
        })

    return not mismatch

def _copy_parameters(source_module, target_module):
    """
    Copy parameters from source PyTorch module to target Triton module.

    Args:
        source_module (torch.nn.Module): Source PyTorch module to copy parameters from.
        target_module (torch.nn.Module): Target Triton module to copy parameters to.

    Note:
        Both modules must have the same parameter names and shapes.
        The copy is performed in-place on the target module's parameters.
    """
    # Iterate over all parameters in the source module
    for name, source_param in source_module.named_parameters():
        # Retrieve the corresponding parameter in the target module by name
        target_param = target_module.get_parameter(name)
        # Copy the data from source to target (in-place)
        target_param.data.copy_(source_param.data)

def _backward_pass(module, inputs, dy):
    """
    Perform backward pass through a module with given inputs and gradients.

    Args:
        module (torch.nn.Module): Neural network module to perform backward pass through.
        inputs (torch.Tensor | Tuple[torch.Tensor]): Input tensor(s) to the module.
        dy (torch.Tensor | Tuple[torch.Tensor]): Gradient tensor(s) for backward pass.

    Returns:
        Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]: Two tuples containing:
            1. Gradients with respect to inputs (dx)
            2. Gradients with respect to module parameters (dw)
    """
    detached_inputs = tuple(x.clone().detach().requires_grad_() for x in inputs) if isinstance(inputs, tuple) else (inputs.clone().detach().requires_grad_(),)
    outputs = module(*detached_inputs)

    if isinstance(outputs, tuple):
        for t_out, t_dy in zip(outputs, dy):
            t_out.backward(t_dy, retain_graph=True)
    else:
        outputs.backward(dy)

    dx, dw = tuple([x.grad for x in detached_inputs]), tuple([p.grad for p in module.parameters()])
    return dx, dw

def _benchmark_implementations(config, triton_fn, torch_fn, inputs, timings_triton, timings_pytorch, speedups, performance, num_warmup=25, num_repeats=100, device='cuda', dtype=torch.float16):
    """
    Benchmark Triton and PyTorch implementations and measure execution times.

    Args:
        config (List[int]): Current configuration being tested (e.g., [batch_size, input_dim, output_dim]).
        triton_fn (Callable): Triton implementation to benchmark.
        torch_fn (Callable): PyTorch implementation to benchmark.
        inputs (torch.Tensor | Tuple[torch.Tensor]): Input tensor(s) for the functions.
        timings_triton (List[float]): List to store Triton execution times.
        timings_pytorch (List[float]): List to store PyTorch execution times.
        speedups (List[float]): List to store speedup ratios (PyTorch/Triton).
        performance (Dict): Dictionary to store detailed performance results.
        num_warmup (int, optional): Number of warmup iterations. Defaults to 25.
        num_repeats (int, optional): Number of timing iterations. Defaults to 100.
        device (str, optional): Device to run on. Defaults to 'cuda'.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float16.

    Returns:
        Tuple[float, float, float]: Tuple containing:
            1. Triton execution time (ms)
            2. PyTorch execution time (ms)
            3. Speedup ratio (PyTorch/Triton)
    """
    quantiles = [0.5]

    # Ensure inputs is a tuple
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # Time Triton implementation
    time_triton = triton.testing.do_bench(
        lambda: triton_fn(inputs),
        warmup=num_warmup,
        rep=num_repeats,
        quantiles=quantiles
    ) * 1e3

    # Time PyTorch implementation
    time_pytorch = triton.testing.do_bench(
        lambda: torch_fn(inputs),
        warmup=num_warmup,
        rep=num_repeats,
        quantiles=quantiles
    ) * 1e3

    timings_triton.append(time_triton)
    timings_pytorch.append(time_pytorch)

    speedup = time_pytorch / time_triton
    speedups.append(speedup)

    print(f"Shape: ({config}, {dtype}), Triton: {time_triton:.1f} us, PyTorch: {time_pytorch:.1f} us, Speedup: {speedup:.2f}x")

    performance["results"].append({
        "shape": config,
        "dtype": str(dtype),
        "triton_time": time_triton,
        "pytorch_time": time_pytorch,
        "speedup": speedup
    })

    return time_triton, time_pytorch, speedup

def _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label=""):
    """
    Plot performance comparison between Triton and PyTorch implementations.

    Args:
        configs (List[List[int]]): List of configurations tested.
        timings_triton (List[float]): List of Triton execution times.
        timings_pytorch (List[float]): List of PyTorch execution times.
        speedups (List[float]): List of speedup ratios (PyTorch/Triton).
        label (str, optional): Label for the plot. Defaults to "".

    Note:
        Saves the plot to 'code.performance.png' and includes:
        - Speedup ratio for each configuration
        - Break-even line at y=1
        - Grid and legend
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x_labels = [f"({config})" for config in configs]
    x = np.arange(len(x_labels))
    width = 0.35

    # Plot timing bars
    ax1.bar(x - width/2, timings_triton, width, label='Triton', color='blue')
    ax1.bar(x + width/2, timings_pytorch, width, label='PyTorch', color='red')

    ax1.set_ylabel('Time (us)')
    ax1.set_xlabel('Tensor Shape')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend(loc='upper left')

    # Plot speedup line
    ax2 = ax1.twinx()
    ax2.plot(x, speedups, label='Speedup (PyTorch / Triton)', color='green', marker='o')
    ax2.set_ylabel('Speedup (x)')
    ax2.legend(loc='upper right')

    filename = f"code.performance.{label}.png"

    fig.tight_layout()
    plt.savefig(filename)
    print(f"Benchmark graph saved to {filename}")
    plt.close(fig)
```)


# text
Fix the code in the code block, original files and error message are provided.
ANALYZE the error message, and fix the code in the code block.
DO NOT change the filename, only fix the code in the code block.
DO NOT add any other files.
DO NOT change the benchmark code.
ONLY respond with the file that contains the fixed code.
MINIMIZE the changes to the code structure.


# text
Error Message: Traceback (most recent call last):
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 339, in <module>
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/benchmark.py", line 66, in verify_correctness_forward
    triton_output = triton_module(*inputs) if isinstance(inputs, tuple) else triton_module(inputs)
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 321, in forward
    return _LinearTriton.apply(x, self.weight, self.bias)
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 274, in forward
    output = linear_triton_forward(x, weight)
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 89, in linear_triton_forward
    linear_forward_kernel[grid](
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 186, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 186, in <dictcomp>
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
  File "/home/centos/triton-cr/venv/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 141, in _bench
    raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
ValueError: Conflicting meta-parameters: GROUP_SIZE_B, GROUP_SIZE_O. Make sure that you don't re-define auto-tuned symbols.


