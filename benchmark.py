# filename: benchmark.py
import torch
import triton
from rmsnorm import RMSNorm, TorchRMSNorm  # Import from the previous file
import time

def benchmark_rmsnorm(M, N, dtype, provider, warmup=25, rep=100):
    if provider == "triton":
        module = RMSNorm(N, dtype=dtype).to(device='cuda')
    elif provider == "torch":
        module = TorchRMSNorm(N, dtype=dtype).to(device='cuda')
    else:
        raise ValueError("Provider must be 'triton' or 'torch'")

    x = torch.randn(M, N, device='cuda', dtype=dtype)

    # Warm-up
    for _ in range(warmup):
        module(x)

    # Benchmark
    if provider == "triton":
        return triton.testing.do_bench(lambda: module(x), warmup=warmup, rep=rep)
    elif provider == "torch":
      start_event = torch.cuda.Event(enable_timing=True)
      end_event = torch.cuda.Event(enable_timing=True)
      start_event.record()
      for _ in range(rep):
          module(x)
      end_event.record()
      torch.cuda.synchronize()  # synchronize here to wait until all kernels launched by matmul are finished.
      return start_event.elapsed_time(end_event)/rep


if __name__ == "__main__":

    # Define sizes and dtypes
    M_values = [128, 256, 512, 1024]
    N_values = [512, 1024, 2048, 4096]
    dtypes = [torch.float32, torch.float16]

    print(f"{'Provider':>8} {'M':>5} {'N':>5} {'dtype':>8} {'Time (ms)':>10}")

    for M in M_values:
      for N in N_values:
          for dtype in dtypes:
            for provider in ["torch", "triton"]:
                latency = benchmark_rmsnorm(M, N, dtype=dtype, provider=provider)
                print(f"{provider:>8} {M:>5} {N:>5} {str(dtype):>8} {latency:>10.4f}")
