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


