# filename: test_linear.py
import torch
import triton
import time
from linear_triton import (
    generated_triton_forward, generated_torch_forward,
    generated_triton_backward, generated_torch_backward
)
from benchmark import verify_correctness, benchmark_performance

def verify_forward_correctness(triton_func, torch_func, input_generator, configs):
    """Verify correctness of forward pass implementation."""
    batch_sizes, in_features, out_features = configs

    print("Verifying forward pass correctness...")
    for batch_size in batch_sizes:
        for in_feat in in_features:
            for out_feat in out_features:
                print(f"Testing batch_size={batch_size}, in_features={in_feat}, out_features={out_feat}")
                # Generate inputs
                x = torch.randn(batch_size, in_feat, device='cuda', dtype=torch.float16)  # [batch_size, in_feat]
                weight = torch.randn(out_feat, in_feat, device='cuda', dtype=torch.float16)  # [out_feat, in_feat]

                # Compute outputs
                triton_output = triton_func(x, weight)  # [batch_size, out_feat]
                torch_output = torch_func(x, weight)  # [batch_size, out_feat]

                # Check correctness
                torch.testing.assert_close(triton_output, torch_output, rtol=1e-2, atol=1e-2)

    print("✓ Forward pass verification passed!")

def verify_backward_correctness(triton_func, torch_func, configs):
    """Verify correctness of backward pass implementation."""
    batch_sizes, in_features, out_features = configs

    print("Verifying backward pass correctness...")
    for batch_size in batch_sizes:
        for in_feat in in_features:
            for out_feat in out_features:
                print(f"Testing batch_size={batch_size}, in_features={in_feat}, out_features={out_feat}")
                # Generate inputs
                x = torch.randn(batch_size, in_feat, device='cuda', dtype=torch.float16)  # [batch_size, in_feat]
                weight = torch.randn(out_feat, in_feat, device='cuda', dtype=torch.float16)  # [out_feat, in_feat]
                grad_output = torch.randn(batch_size, out_feat, device='cuda', dtype=torch.float16)  # [batch_size, out_feat]

                # Compute outputs
                triton_grad_x, triton_grad_weight = triton_func(grad_output, x, weight)  # [batch_size, in_feat], [out_feat, in_feat]
                torch_grad_x, torch_grad_weight = torch_func(grad_output, x, weight)  # [batch_size, in_feat], [out_feat, in_feat]

                # Check correctness
                torch.testing.assert_close(triton_grad_x, torch_grad_x, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(triton_grad_weight, torch_grad_weight, rtol=1e-2, atol=1e-2)

    print("✓ Backward pass verification passed!")

def benchmark_forward_performance(triton_func, torch_func, configs):
    """Benchmark performance of forward pass implementation."""
    batch_sizes, in_features, out_features = configs

    print("\nBenchmarking forward pass performance...")
    print(f"{'Batch Size':<10}{'In Features':<12}{'Out Features':<12}{'PyTorch (ms)':<15}{'Triton (ms)':<15}{'Speedup':<10}")

    for batch_size in batch_sizes:
        for in_feat in in_features:
            for out_feat in out_features:
                # Generate inputs
                x = torch.randn(batch_size, in_feat, device='cuda', dtype=torch.float16)  # [batch_size, in_feat]
                weight = torch.randn(out_feat, in_feat, device='cuda', dtype=torch.float16)  # [out_feat, in_feat]

                # Warmup
                for _ in range(10):
                    triton_func(x, weight)
                    torch_func(x, weight)

                torch.cuda.synchronize()

                # Benchmark PyTorch
                torch_times = []
                for _ in range(20):
                    start = time.time()
                    torch_func(x, weight)
                    torch.cuda.synchronize()
                    torch_times.append((time.time() - start) * 1000)  # ms

                # Benchmark Triton
                triton_times = []
                for _ in range(20):
                    start = time.time()
                    triton_func(x, weight)
                    torch.cuda.synchronize()
                    triton_times.append((time.time() - start) * 1000)  # ms

                avg_torch = sum(torch_times) / len(torch_times)
                avg_triton = sum(triton_times) / len(triton_times)
                speedup = avg_torch / avg_triton if avg_triton > 0 else float('inf')

                print(f"{batch_size:<10}{in_feat:<12}{out_feat:<12}{avg_torch:<15.3f}{avg_triton:<15.3f}{speedup:<10.3f}")

def benchmark_backward_performance(triton_func, torch_func, configs):
    """Benchmark performance of backward pass implementation."""
    batch_sizes, in_features, out_features = configs

    print("\nBenchmarking backward pass performance...")
    print(f"{'Batch Size':<10}{'In Features':<12}{'Out Features':<12}{'PyTorch (ms)':<15}{'Triton (ms)':<15}{'Speedup':<10}")

    for batch_size in batch_sizes:
        for in_feat in in_features:
            for out_feat in out_features:
                # Generate inputs
                x = torch.randn(batch_size, in_feat, device='cuda', dtype=torch.float16)  # [batch_size, in_feat]
                weight = torch.randn(out_feat, in_feat, device='cuda', dtype=torch.float16)  # [out_feat, in_feat]
                grad_output = torch.randn(batch_size, out_feat, device='cuda', dtype=torch.float16)  # [batch_size, out_feat]

                # Warmup
                for _ in range(10):
                    triton_func(grad_output, x, weight)
                    torch_func(grad_output, x, weight)

                torch.cuda.synchronize()

                # Benchmark PyTorch
                torch_times = []
                for _ in range(20):
                    start = time.time()
                    torch_func(grad_output, x, weight)
                    torch.cuda.synchronize()
                    torch_times.append((time.time() - start) * 1000)  # ms

                # Benchmark Triton
                triton_times = []
                for _ in range(20):
                    start = time.time()
                    triton_func(grad_output, x, weight)
                    torch.cuda.synchronize()
                    triton_times.append((time.time() - start) * 1000)  # ms

                avg_torch = sum(torch_times) / len(torch_times)
                avg_triton = sum(triton_times) / len(triton_times)
                speedup = avg_torch / avg_triton if avg_triton > 0 else float('inf')

                print(f"{batch_size:<10}{in_feat:<12}{out_feat:<12}{avg_torch:<15.3f}{avg_triton:<15.3f}{speedup:<10.3f}")

if __name__ == "__main__":
    # Test configurations
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    in_features = [128, 256, 512, 1024, 2048]
    out_features = [128, 256, 512, 1024, 2048]

    configs_small = [
        [32, 64, 128],
        [128, 256, 512],
        [128, 256, 512]
    ]

    configs_full = [batch_sizes, in_features, out_features]

    # Verify correctness (using smaller set for speed)
    verify_forward_correctness(
        generated_triton_forward,
        generated_torch_forward,
        None,  # Not used in this implementation
        configs_small
    )

    verify_backward_correctness(
        generated_triton_backward,
        generated_torch_backward,
        configs_small
    )

    # Benchmark performance (with more configurations)
    benchmark_forward_performance(
        generated_triton_forward,
        generated_torch_forward,
        configs_full
    )

    benchmark_backward_performance(
        generated_triton_backward,
        generated_torch_backward,
        configs_full
    )
