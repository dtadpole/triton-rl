import torch
import triton
import triton.language as tl
import time

# --- Triton Kernel (from previous response) ---
@triton.jit
def rmsnorm_kernel(
    input_ptr, output_ptr, gamma_ptr,
    input_row_stride, input_col_stride, gamma_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets * input_col_stride
    gamma_ptrs = gamma_ptr + col_offsets * gamma_stride
    input_values = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)
    gamma_values = tl.load(gamma_ptrs, mask=col_offsets < n_cols, other=1.0)
    squares = input_values * input_values
    mean_squares = tl.sum(squares, axis=0) / n_cols
    rms = tl.sqrt(mean_squares + eps)
    normalized_values = input_values / rms
    output_values = normalized_values * gamma_values
    output_ptrs = output_ptr + row_idx * input_row_stride + col_offsets * input_col_stride
    tl.store(output_ptrs, output_values, mask=col_offsets < n_cols)

def rmsnorm(input: torch.Tensor, gamma: torch.Tensor, eps: float):
    if input.ndim != 2:
      raise ValueError("Input must have 2 dimensions: (rows, features)")
    n_rows, n_cols = input.shape
    output = torch.empty_like(input)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16
    grid = (n_rows,)
    rmsnorm_kernel[grid](
        input, output, gamma,
        input.stride(0), input.stride(1), gamma.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

# --- PyTorch Implementation (from previous response) ---
def rmsnorm_torch(input: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)
    return (input / rms) * gamma

# --- Performance Test Function ---
def benchmark_rmsnorm(rows, cols, eps, dtype, device, n_runs=100):
    # Create input tensor
    torch.manual_seed(0)
    input = torch.randn((rows, cols), dtype=dtype, device=device)
    gamma = torch.randn((cols,), dtype=dtype, device=device)

    # Warm-up (important for accurate timing, especially on GPU)
    for _ in range(10):
        _ = rmsnorm(input.clone(), gamma.clone(), eps)
        _ = rmsnorm_torch(input.clone(), gamma.clone(), eps)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Triton timing
    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else: # device == cpu
      start_time = time.time()
    for _ in range(n_runs):
        output_triton = rmsnorm(input.clone(), gamma.clone(), eps)
    if device == 'cuda':
      end_event.record()
      torch.cuda.synchronize()
      triton_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to s
    else:
      end_time = time.time()
      triton_time = end_time - start_time

    # PyTorch timing
    if device == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
      start_time = time.time()
    for _ in range(n_runs):
        output_torch = rmsnorm_torch(input.clone(), gamma.clone(), eps)
    if device == 'cuda':
      end_event.record()
      torch.cuda.synchronize()
      torch_time = start_event.elapsed_time(end_event) / 1000  # Convert ms to s
    else:
      end_time = time.time()
      torch_time = end_time - start_time

    # Average time and throughput
    triton_time /= n_runs
    torch_time /= n_runs

    # Calculate throughput in GB/s
    element_size = input.element_size()
    triton_throughput = (rows * cols * element_size * 2) / triton_time / 1e9  # *2 for read and write
    torch_throughput = (rows * cols * element_size * 2) / torch_time / 1e9  # *2 for read and write

    print(f"Shape: ({rows}, {cols}), dtype: {dtype}, device: {device}")
    print(f"  Triton time: {triton_time:.6f} s, Throughput: {triton_throughput:.3f} GB/s")
    print(f"  PyTorch time: {torch_time:.6f} s, Throughput: {torch_throughput:.3f} GB/s")
    print(f"  Speedup (Triton/PyTorch): {torch_time / triton_time:.3f}x")
    return triton_time, torch_time



# --- Performance Test Cases ---
performance_test_cases = [
    (1024, 1024, 1e-5, torch.float32, 'cuda'),
    (2048, 2048, 1e-5, torch.float32, 'cuda'),
    (4096, 4096, 1e-5, torch.float32, 'cuda'),
    (1024, 1024, 1e-5, torch.float16, 'cuda'),
    (2048, 2048, 1e-5, torch.float16, 'cuda'),
    (4096, 4096, 1e-5, torch.float16, 'cuda'),
    (1024, 1024, 1e-5, torch.float32, 'cpu'),  # CPU tests
    (2048, 2048, 1e-5, torch.float32, 'cpu'),
    (1024, 1024, 1e-5, torch.float16, 'cpu'),
    (2048, 2048, 1e-5, torch.float16, 'cpu'),
]

for rows, cols, eps, dtype, device in performance_test_cases:
    benchmark_rmsnorm(rows, cols, eps, dtype, device)

print("Performance tests completed.")
