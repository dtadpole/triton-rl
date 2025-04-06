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
from torch.cuda.amp import custom_fwd, custom_bwd

class TritonLinear(torch.nn.Module):
    """
    A PyTorch module implementing a linear layer using Triton kernels for forward and backward passes.
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        device (str): Computation device ('cuda' or 'cpu').
        dtype (torch.dtype): Data type for parameters and computations.
    """
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight):
        """
        Forward pass of the Triton linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, I).
            weight (torch.Tensor): Weight tensor of shape (O, I).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, O).
        """
        ctx.save_for_backward(x, weight)
        B, I = x.shape
        O = weight.shape[0]
        output = torch.empty((B, O), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(O, meta['BLOCK_SIZE_O']))
        forward_kernel[grid](
            x, weight, output,
            B, I, O,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass of the Triton linear layer.
        
        Args:
            grad_output (torch.Tensor): Gradient of output tensor of shape (B, O).
            
        Returns:
            Tuple[torch.Tensor]: Gradients for input and weight tensors.
        """
        x, weight = ctx.saved_tensors
        B, I = x.shape
        O = weight.shape[0]

        grad_input = torch.empty_like(x)
        grad_weight = torch.empty_like(weight)

        grid_dx = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        backward_input_kernel[grid_dx](
            grad_output, weight, grad_input,
            B, O, I,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        grid_dw = lambda meta: (triton.cdiv(O, meta['BLOCK_SIZE_O']), triton.cdiv(I, meta['BLOCK_SIZE_I']))
        backward_weight_kernel[grid_dw](
            x, grad_output, grad_weight,
            B, I, O,
            x.stride(0), x.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )

        return grad_input, grad_weight

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 128}, num_warps=16),
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
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
):
    """Triton kernel for forward pass of linear layer."""
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_O), dtype=tl.float32)
    
    for i in range(0, I, BLOCK_SIZE_I):
        i_offs = i + tl.arange(0, BLOCK_SIZE_I)
        mask_i = i_offs < I
        
        input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + i_offs[None, :] * stride_input_i
        weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + i_offs[None, :] * stride_weight_i
        
        mask_input = (offs_b[:, None] < B) & mask_i
        input = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        mask_weight = (offs_o[:, None] < O) & mask_i
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(input, weight, trans_b=True, allow_tf32=True)
    
    mask_output = (offs_b[:, None] < B) & (offs_o[None, :] < O)
    tl.store(output_ptr + offs_b[:, None] * stride_output_b + offs_o[None, :] * stride_output_o,
             acc.to(tl.float16), mask=mask_output)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_O': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_O': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_O': 128}, num_warps=16),
    ],
    key=['B', 'O', 'I']
)
@triton.jit
def backward_input_kernel(
    d_output_ptr, weight_ptr, grad_input_ptr,
    B, O, I,
    stride_dout_b, stride_dout_o,
    stride_weight_o, stride_weight_i,
    stride_grad_b, stride_grad_i,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_O: tl.constexpr,
):
    """Triton kernel for input gradient computation."""
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
    
    for o in range(0, O, BLOCK_SIZE_O):
        o_offs = o + tl.arange(0, BLOCK_SIZE_O)
        mask_o = o_offs < O
        
        dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + o_offs[None, :] * stride_dout_o
        weight_ptrs = weight_ptr + o_offs[:, None] * stride_weight_o + offs_i[None, :] * stride_weight_i
        
        mask_dout = (offs_b[:, None] < B) & mask_o
        dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        
        mask_weight = mask_o[:, None] & (offs_i[None, :] < I)
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)
        
        acc += tl.dot(dout, weight, allow_tf32=True)
    
    mask_grad = (offs_b[:, None] < B) & (offs_i[None, :] < I)
    tl.store(grad_input_ptr + offs_b[:, None] * stride_grad_b + offs_i[None, :] * stride_grad_i,
             acc.to(tl.float16), mask=mask_grad)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_O': 64, 'BLOCK_SIZE_I': 64, 'BLOCK_SIZE_B': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_O': 128, 'BLOCK_SIZE_I': 128, 'BLOCK_SIZE_B': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_O': 256, 'BLOCK_SIZE_I': 256, 'BLOCK_SIZE_B': 128}, num_warps=16),
    ],
    key=['O', 'I', 'B']
)
@triton.jit
def backward_weight_kernel(
    input_ptr, d_output_ptr, grad_weight_ptr,
    B, I, O,
    stride_input_b, stride_input_i,
    stride_dout_b, stride_dout_o,
    stride_grad_o, stride_grad_i,
    BLOCK_SIZE_O: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
):
    """Triton kernel for weight gradient computation."""
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    offs_o = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    
    acc = tl.zeros((BLOCK_SIZE_O, BLOCK_SIZE_I), dtype=tl.float32)
    
    for b in range(0, B, BLOCK_SIZE_B):
        b_offs = b + tl.arange(0, BLOCK_SIZE_B)
        mask_b = b_offs < B
        
        input_ptrs = input_ptr + b_offs[:, None] * stride_input_b + offs_i[None, :] * stride_input_i
        dout_ptrs = d_output_ptr + b_offs[:, None] * stride_dout_b + offs_o[None, :] * stride_dout_o
        
        mask_input = mask_b[:, None] & (offs_i[None, :] < I)
        input = tl.load(input_ptrs, mask=mask_input, other=0.0)
        
        mask_dout = mask_b[:, None] & (offs_o[None, :] < O)
        dout = tl.load(dout_ptrs, mask=mask_dout, other=0.0)
        
        acc += tl.dot(tl.trans(dout), input, allow_tf32=True)
    
    mask_grad = (offs_o[:, None] < O) & (offs_i[None, :] < I)
    tl.store(grad_weight_ptr + offs_o[:, None] * stride_grad_o + offs_i[None, :] * stride_grad_i,
             acc.to(tl.float16), mask=mask_grad)

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
Error Message: /home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/01/linear_triton.py:31: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd
/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/01/linear_triton.py:58: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd
Traceback (most recent call last):
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/language/core.py", line 35, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
TypeError: dot() got an unexpected keyword argument 'trans_b'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/01/linear_triton.py", line 247, in <module>
    verify_correctness_forward(module_generator, forward_generator, [B_sizes, I_sizes, O_sizes])
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/01/benchmark.py", line 66, in verify_correctness_forward
    triton_output = triton_module(*inputs) if isinstance(inputs, tuple) else triton_module(inputs)
                                                                             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/01/linear_triton.py", line 27, in forward
    return TritonLinearFunction.apply(x, self.weight)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/torch/amp/autocast_mode.py", line 503, in decorate_fwd
    return fwd(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/module_linear/code_gen_deepseek-fw_002/01/linear_triton.py", line 48, in forward
    forward_kernel[grid](
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 186, in run
    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 166, in _bench
    return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/testing.py", line 117, in do_bench
    fn()
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 152, in kernel_call
    self.fn.run(
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
             ^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/compiler/compiler.py", line 273, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/centos/triton-rl/.venv/lib/python3.12/site-packages/triton/compiler/compiler.py", line 100, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 31:15:
        mask_i = i_offs < I

        input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + i_offs[None, :] * stride_input_i
        weight_ptrs = weight_ptr + offs_o[:, None] * stride_weight_o + i_offs[None, :] * stride_weight_i

        mask_input = (offs_b[:, None] < B) & mask_i
        input = tl.load(input_ptrs, mask=mask_input, other=0.0)

        mask_weight = (offs_o[:, None] < O) & mask_i
        weight = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        acc += tl.dot(input, weight, trans_b=True, allow_tf32=True)
               ^


