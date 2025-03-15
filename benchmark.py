# filename: benchmark.py

import torch
import triton
import triton.testing
import numpy as np
import matplotlib.pyplot as plt
import json

# dims is a list of possible dimensions
# e.g. dims = [[16, 32], [128, 256, 512], [256, 512, 1024, 2048]]
def _generate_configs(dims):
    """
    Generate all possible configurations of the dimensions.
    e.g. dims = [[16, 32], [128, 256], [512, 1024]]
    returns: [
        [16, 128, 512],
        [16, 128, 1024],
        [16, 256, 512],
        [16, 256, 1024],
        [32, 128, 512],
        [32, 128, 1024],
        [32, 256, 512],
        [32, 256, 1024],
    ]
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
    # Iterate over all parameters in the source module
    for name, source_param in source_module.named_parameters():
        # Retrieve the corresponding parameter in the target module by name
        target_param = target_module.get_parameter(name)
        # Copy the data from source to target (in-place)
        target_param.data.copy_(source_param.data)


def _backward_pass(module, inputs, dy):
    detached_inputs = tuple(x.clone().detach().requires_grad_() for x in inputs) if isinstance(inputs, tuple) else (inputs.clone().detach().requires_grad_(),)
    outputs = module(*detached_inputs)

    if isinstance(outputs, tuple):
        for t_out, t_dy in zip(outputs, dy):
            t_out.backward(t_dy, retain_graph=True)
    else:
        outputs.backward(dy)

    dx, dw = tuple([x.grad for x in detached_inputs]), tuple([p.grad for p in module.parameters()])
    return dx, dw


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


def _benchmark_implementations(config, triton_fn, torch_fn, inputs, timings_triton, timings_pytorch, speedups, performance, num_warmup=25, num_repeats=100, device='cuda', dtype=torch.float16):
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

        time_triton, time_pytorch, speedup = _benchmark_implementations(
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

        time_triton, time_pytorch, speedup = _benchmark_implementations(
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

        time_triton, time_pytorch, speedup = _benchmark_implementations(
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


if __name__ == '__main__':
    # Correctness Test
    batch_sizes = [16, 32, 64]
    dimensions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # print(generate_configs([batch_sizes, dimensions, [0, 1, 2]]))
    correctness = verify_correctness_func(lambda x: torch.zeros(batch_sizes[0], dimensions[0]), lambda x: torch.zeros(batch_sizes[0], dimensions[0]), [batch_sizes, dimensions])
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    # Performance Test
    batch_sizes = [64, 128, 256]
    dimensions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    median_triton, median_pytorch, speedup, performance = benchmark_performance_func(lambda x: torch.zeros(batch_sizes[0], dimensions[0]), lambda x: torch.zeros(batch_sizes[0], dimensions[0]), [batch_sizes, dimensions])
    print(f"Median performance - Triton: {median_triton:.4f} ms, PyTorch: {median_pytorch:.4f} ms, Speedup: {speedup:.2f}x")
