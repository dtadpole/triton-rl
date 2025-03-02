# filename: test_softmax.py
import torch
from benchmark import verify_correctness, benchmark_performance
from softmax_implementation import generated_triton_func, generated_torch_func

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x

    print("Verifying correctness...")
    verify_correctness(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])

    print("Benchmarking performance...")
    benchmark_performance(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
