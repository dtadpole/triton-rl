import torch
import triton
import triton.testing
import numpy as np
import matplotlib.pyplot as plt


# dims is a list of possible dimensions
# e.g. dims = [[16, 32], [128, 256, 512], [256, 512, 1024, 2048]]
def generate_configs(dims):
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


def verify_correctness(func_triton, func_torch, input_generator, dimensions, device='cuda', dtype=torch.float16, atol=1e-2, rtol=1e-2, seed=42):
    """
    Verifies the correctness of the Triton implementation against PyTorch.
    """
    torch.manual_seed(seed)
    passed = 0
    total = 0

    for config in generate_configs(dimensions):
        inputs = input_generator(*config, device=device, dtype=dtype)
        if isinstance(inputs, tuple):
            triton_output = func_triton(*inputs)
            torch_output = func_torch(*inputs)
        else:
            triton_output = func_triton(inputs)
            torch_output = func_torch(inputs)

        if torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
            print(f"✅ Correctness test passed for shape ({config}, {dtype})")
            passed += 1
        else:
            print(f"❌ Correctness test failed for shape ({config}, {dtype})")
        total += 1

    return passed, total


def benchmark_performance(func_triton, func_torch, input_generator, dimensions, num_warmup=10, num_repeats=100, device='cuda', dtype=torch.float16):
    """
    Benchmarks the performance of the Triton implementation against PyTorch.
    """
    torch.manual_seed(0)

    configs = generate_configs(dimensions)
    quantiles = [0.5]

    pytorch_compiled = torch.compile(func_torch, mode="max-autotune")

    timings_triton = []
    timings_pytorch = []
    speedups = []

    for config in configs:
        inputs = input_generator(*config, device=device, dtype=dtype)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        # Triton timing
        time_triton = triton.testing.do_bench(lambda: func_triton(*inputs), warmup=num_warmup, rep=num_repeats, quantiles=quantiles) * 1e3
        timings_triton.append(time_triton)

        # PyTorch timing
        time_pytorch = triton.testing.do_bench(lambda: pytorch_compiled(*inputs), warmup=num_warmup, rep=num_repeats, quantiles=quantiles) * 1e3
        timings_pytorch.append(time_pytorch)

        speedup = time_pytorch / time_triton
        speedups.append(speedup)

        print(f"Shape: ({config}, {dtype}), Triton: {time_triton:.1f} us, PyTorch: {time_pytorch:.1f} us, Speedup: {speedup:.2f}x")

    # Graphing
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x_labels = [f"({config})" for config in configs]
    x = np.arange(len(x_labels))
    width = 0.35

    ax1.bar(x - width/2, timings_triton, width, label='Triton', color='blue')
    ax1.bar(x + width/2, timings_pytorch, width, label='PyTorch', color='red')

    ax1.set_ylabel('Time (us)')
    ax1.set_xlabel('Tensor Shape')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(x, speedups, label='Speedup (PyTorch / Triton)', color='green', marker='o')
    ax2.set_ylabel('Speedup (x)')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    graph_filename = "benchmark.png"
    plt.savefig(graph_filename)
    print(f"Benchmark graph saved to {graph_filename}")

    median_time_triton = np.median(timings_triton)
    median_time_pytorch = np.median(timings_pytorch)
    median_speedup = np.median(speedups)

    return median_time_triton, median_time_pytorch, median_speedup


if __name__ == '__main__':
    # Correctness Test
    batch_sizes = [16, 32, 64]
    dimensions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # print(generate_configs([batch_sizes, dimensions, [0, 1, 2]]))
    correct, total = verify_correctness(lambda x: torch.zeros(batch_sizes[0], dimensions[0]), lambda x: torch.zeros(batch_sizes[0], dimensions[0]), [batch_sizes, dimensions])
    print(f"Correctness tests: {correct}/{total} passed")

    # Performance Test
    batch_sizes = [64, 128, 256]
    dimensions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    median_triton, median_pytorch, speedup = benchmark_performance(lambda x: torch.zeros(batch_sizes[0], dimensions[0]), lambda x: torch.zeros(batch_sizes[0], dimensions[0]), [batch_sizes, dimensions])
    print(f"Median performance - Triton: {median_triton:.4f} ms, PyTorch: {median_pytorch:.4f} ms, Speedup: {speedup:.2f}x")
