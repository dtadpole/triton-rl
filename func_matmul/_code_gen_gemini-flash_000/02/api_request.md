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
Generate triton code for Matrix Multiplication (filename: matmul_triton.py)
    use autotune for block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), use super blocks and autotune GROUP_SIZE_M,
    test correctness and performance using following code literally:
```python
if __name__ == "__main__":
    M_sizes = [256, 1024, 4096]
    N_sizes = [256, 1024, 4096]
    K_sizes = [256, 512, 1024, 2048, 4096]
    def input_generator(M, N, K, device='cuda', dtype=torch.float16):
        A = torch.randn(M, K, device=device, dtype=dtype)
        A = torch.nn.functional.normalize(A, dim=-1)
        B = torch.randn(K, N, device=device, dtype=dtype)
        B = torch.nn.functional.normalize(B, dim=-1)
        return A, B

    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
    benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [M_sizes, N_sizes, K_sizes])
```


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
    with open("code.correctness.json", "w") as f:
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
    pytorch_compiled = torch.compile(func_torch, mode="max-autotune")

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


