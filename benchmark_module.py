# filename: benchmark.py
import json
import torch
import torch.nn as nn
import triton
import triton.testing
import numpy as np
import matplotlib.pyplot as plt


# dims is a list of possible dimensions
# e.g. dims = [[16, 32], [128, 256, 512], [256, 512, 1024, 2048]]
def generate_configs(dims):
    """
    Generate all possible configurations from the dimensions.
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


def verify_backward(TritonModule, TorchModule):
    # torch.manual_seed(0)
    in_features = 512
    out_features = 2048
    batch_size = 768
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float16)
    x = nn.functional.normalize(x, dim=-1)

    linear_torch = nn.Linear(in_features, out_features, bias=False, device='cuda', dtype=torch.float16)
    linear_triton = TritonModule(in_features, out_features, device='cuda', dtype=torch.float16)
    linear_triton.reset_parameters(linear_torch.weight)

    dy = torch.randn(batch_size, out_features, device='cuda', dtype=torch.float16)
    dy = torch.nn.functional.normalize(dy, dim=1)

    # Triton backward
    x_triton = x.clone().detach().requires_grad_()
    y_triton = linear_triton(x_triton)
    y_triton.backward(dy)
    dx_triton, dw_triton = x_triton.grad, linear_triton.weight.grad

    # PyTorch backward
    x_pytorch = x.clone().detach().requires_grad_()
    y_pytorch = linear_torch(x_pytorch)
    y_pytorch.backward(dy)
    dx_pytorch, dw_pytorch = x_pytorch.grad, linear_torch.weight.grad

    if torch.allclose(dx_triton, dx_pytorch, rtol=1e-2, atol=1e-2):
        print("✅ Backward dx grad passed!")
    else:
        print("❌ Backward dx grad failed!")
    if torch.allclose(dw_pytorch, dw_triton, rtol=1e-2, atol=1e-2):
        print("✅ Backward dw grad passed!")
    else:
        print("❌ Backward dw grad failed!")

def benchmark(device='cuda', forward=True, backward=True):
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    for (M, N, K) in sizes:
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        weight = torch.randn(N, K, device=device, dtype=torch.float16)
        dy = torch.randn(M, N, device=device, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        linear_torch = torch.nn.Linear(K, N, bias=False, device='cuda', dtype=torch.float16)
        linear_triton = LinearTriton(K, N, device='cuda', dtype=torch.float16)

        if forward:
            print(f'Forward pass - {M}x{K} @ {N}x{K}^T -> {M}x{N}')
            triton_time = triton.testing.do_bench(lambda: linear_triton(x), quantiles=quantiles)
            torch_time = triton.testing.do_bench(lambda: linear_torch(x), quantiles=quantiles)
            print(f'Triton: {triton_time[0]:.3f} ms')
            print(f'PyTorch: {torch_time[0]:.3f} ms')
            print(f'Speedup: {torch_time[0]/triton_time[0]:.2f}x')

        if backward:
            print(f'Backward pass - dX: {M}x{N} @ {N}x{K} -> {M}x{K}, dW: {N}x{M} @ {M}x{K} -> {N}x{K}')
            x_triton = x.clone().detach().requires_grad_()
            x_torch = x.clone().detach().requires_grad_()
            def triton_backward():
                y_triton = linear_triton(x_triton)
                y_triton.backward(dy)
                # dx_triton, dw_triton = x_triton.grad, linear_triton.weight.grad
            def torch_backward():
                y_torch = linear_torch(x_torch)
                y_torch.backward(dy)
                # dx_torch, dw_torch = x_torch.grad, linear_torch.weight.grad
            triton_time = triton.testing.do_bench(triton_backward, quantiles=quantiles)
            torch_time = triton.testing.do_bench(torch_backward, quantiles=quantiles)
            print(f'Triton: {triton_time[0]:.3f} ms')
            print(f'PyTorch: {torch_time[0]:.3f} ms')
            print(f'Speedup: {torch_time[0]/triton_time[0]:.2f}x')


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
