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
    """
    Compute forward pass for linear layer using block matrix multiplication.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor of shape (B, I)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (O, I)
        output_ptr (tl.tensor): Pointer to output tensor of shape (B, O)
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_ib (int): Input batch dimension stride
        stride_ii (int): Input input dimension stride
        stride_wo (int): Weight output dimension stride
        stride_wi (int): Weight input dimension stride
        stride_ob (int): Output batch dimension stride
        stride_oo (int): Output output dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_M (tl.constexpr): Number of groups for super-blocking

    Explanation:
        1. Calculate program IDs and block offsets
        2. Load input and weight blocks using matrix pointers
        3. Compute matrix multiplication using dot product
        4. Store results to output tensor
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(O, BLOCK_SIZE_O)
    num_pid_k = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_b = pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = tl.arange(0, BLOCK_SIZE_I)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for k in range(0, num_pid_k):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] < I)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input_block, weight_block, allow_tf32=True)
        
        input_ptrs += BLOCK_SIZE_I * stride_ii
        weight_ptrs += BLOCK_SIZE_I * stride_wi
    
    mask_output = (offs_b[:, None] < B) & (offs_o[None, :] < O)
    output_ptrs = output_ptr + offs_b[:, None] * stride_ob + offs_o[None, :] * stride_oo
    tl.store(output_ptrs, acc.to(tl.float16), mask=mask_output)


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
    """
    Compute gradient of input for linear layer using matrix multiplication.

    Args:
        d_output_ptr (tl.tensor): Pointer to output gradient tensor of shape (B, O)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (O, I)
        d_input_ptr (tl.tensor): Pointer to input gradient tensor of shape (B, I)
        B (int): Batch dimension size
        O (int): Output dimension size
        I (int): Input dimension size
        stride_dob (int): Output gradient batch dimension stride
        stride_doo (int): Output gradient output dimension stride
        stride_wo (int): Weight output dimension stride
        stride_wi (int): Weight input dimension stride
        stride_dib (int): Input gradient batch dimension stride
        stride_dii (int): Input gradient input dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_O (tl.constexpr): Block size for output dimension
        GROUP_SIZE_M (tl.constexpr): Number of groups for super-blocking

    Explanation:
        1. Calculate program IDs and block offsets
        2. Load output gradient and weight blocks
        3. Compute matrix multiplication with weight transpose
        4. Store results to input gradient tensor
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(I, BLOCK_SIZE_O)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_b = pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_o = tl.arange(0, BLOCK_SIZE_O)
    
    d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo
    weight_ptrs = weight_ptr + offs_o[:, None] * stride_wo + offs_i[None, :] * stride_wi
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(O, BLOCK_SIZE_O)):
        mask_dout = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        mask_weight = (offs_o[:, None] < O) & (offs_i[None, :] < I)
        
        d_output_block = tl.load(d_output_ptrs, mask=mask_dout, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(d_output_block, weight_block, allow_tf32=True)
        
        d_output_ptrs += BLOCK_SIZE_O * stride_doo
        weight_ptrs += BLOCK_SIZE_O * stride_wo
    
    mask_din = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    d_input_ptrs = d_input_ptr + offs_b[:, None] * stride_dib + offs_i[None, :] * stride_dii
    tl.store(d_input_ptrs, acc.to(tl.float16), mask=mask_din)


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
    """
    Compute gradient of weight for linear layer using outer product.

    Args:
        input_ptr (tl.tensor): Pointer to input tensor of shape (B, I)
        d_output_ptr (tl.tensor): Pointer to output gradient tensor of shape (B, O)
        d_weight_ptr (tl.tensor): Pointer to weight gradient tensor of shape (O, I)
        B (int): Batch dimension size
        I (int): Input dimension size
        O (int): Output dimension size
        stride_ib (int): Input batch dimension stride
        stride_ii (int): Input input dimension stride
        stride_dob (int): Output gradient batch dimension stride
        stride_doo (int): Output gradient output dimension stride
        stride_dwo (int): Weight gradient output dimension stride
        stride_dwi (int): Weight gradient input dimension stride
        BLOCK_SIZE_B (tl.constexpr): Block size for batch dimension
        BLOCK_SIZE_I (tl.constexpr): Block size for input dimension
        GROUP_SIZE_M (tl.constexpr): Number of groups for super-blocking

    Explanation:
        1. Calculate program IDs and block offsets
        2. Load input and output gradient blocks
        3. Compute outer product accumulation
        4. Atomically store results to weight gradient tensor
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(O, BLOCK_SIZE_B)
    num_pid_n = tl.cdiv(I, BLOCK_SIZE_I)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_o = pid_m * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_n * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offs_b = tl.arange(0, BLOCK_SIZE_B)
    
    input_ptrs = input_ptr + offs_b[:, None] * stride_ib + offs_i[None, :] * stride_ii
    d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dob + offs_o[None, :] * stride_doo
    
    acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_B), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        mask_input = (offs_b[:, None] < B) & (offs_i[None, :] < I)
        mask_dout = (offs_b[:, None] < B) & (offs_o[None, :] < O)
        
        input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)
        d_output_block = tl.load(d_output_ptrs, mask=mask_dout, other=0.0)
        
        acc += tl.dot(input_block.T, d_output_block, allow_tf32=True)
        
        input_ptrs += BLOCK_SIZE_B * stride_ib
        d_output_ptrs += BLOCK_SIZE_B * stride_dob
    
    mask_dw = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    d_weight_ptrs = d_weight_ptr + offs_o[:, None] * stride_dwo + offs_i[None, :] * stride_dwi
    tl.atomic_add(d_weight_ptrs, acc.T.to(tl.float16), mask=mask_dw)


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
            GROUP_SIZE_M=8
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
            GROUP_SIZE_M=8
        )
        
        # Compute weight gradient
        grid_weight = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),)
        _backward_weight_kernel[grid_weight](
            input, d_output, d_weight,
            B, I, O,
            input.stride(0), input.stride(1),
            d_output.stride(0), d_output.stride(1),
            d_weight.stride(0), d_weight.stride(1),
            GROUP_SIZE_M=8
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
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 332, in <module>
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/benchmark.py", line 66, in verify_correctness_forward
    triton_output = triton_module(*inputs) if isinstance(inputs, tuple) else triton_module(inputs)
                                                                             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 313, in forward
    return TritonLinearFunction.apply(x, self.weight)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/00/linear_triton.py", line 260, in forward
    _forward_kernel[grid](
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 186, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 141, in _bench
    raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
ValueError: Conflicting meta-parameters: GROUP_SIZE_M. Make sure that you don't re-define auto-tuned symbols.


