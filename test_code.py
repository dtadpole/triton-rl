import torch
import triton
import triton.language as tl
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================= Triton Kernel with Autotuning =============================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],  # Specify which arguments to use as keys
)
@triton.jit
def matmul_kernel_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_stride_row, a_stride_col,
    b_stride_row, b_stride_col,
    c_stride_row, c_stride_col,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Optimized Triton kernel (Implementation from the previous detailed response)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_n = tl.arange(0, BLOCK_SIZE_N)

    mask_m = (offset_m + block_start_m) < M
    mask_n = (offset_n + block_start_n) < N

    a_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    b_tile = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_load_ptr = a_ptr + (block_start_m + offset_m[:, None]) * a_stride_row + (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * a_stride_col
        b_load_ptr = b_ptr + (k + offset_m[:, None]) * b_stride_row + (block_start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * b_stride_col

        a_mask = mask_m[:, None] & (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < K
        b_mask = (k + offset_m[:, None]) < K & mask_n[None, :]

        a_tile = tl.load(a_load_ptr, mask=a_mask, other=0.0)
        b_tile = tl.load(b_load_ptr, mask=b_mask, other=0.0)

        # tl.synchronize()

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # Reset per K iteration
        accumulator += tl.dot(a_tile, b_tile)

        # tl.synchronize()

    c_write_ptr = c_ptr + (block_start_m + offset_m[:, None]) * c_stride_row + (block_start_n + tl.arange(0, BLOCK_SIZE_N)[None, :]) * c_stride_col
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_write_ptr, accumulator, mask=c_mask)

def triton_matmul_optimized(a, b):
    M, K = a.shape
    K, N = b.shape
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # No need to define BLOCK_SIZE etc. here; autotune will handle them
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel_optimized[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
# ============================= Benchmarking Framework =============================

def generate_test_data(M, N, K, data_type, device):
    """Generates random matrices A (M x K) and B (K x N) of the specified data type and device."""
    a = torch.randn((M, K), dtype=data_type, device=device)
    b = torch.randn((K, N), dtype=data_type, device=device)
    return a, b

def benchmark_matmul(M, N, K, data_type, device, num_runs=10):
    """
    Benchmarks both the Triton kernel and PyTorch's native matrix multiplication.

    Args:
        M, N, K (int): Matrix dimensions.
        data_type (torch.dtype): Data type for the matrices.
        device (str): Device to run the benchmark on ('cuda' or 'cpu').
        num_runs (int): Number of benchmark iterations.

    Returns:
        dict: Dictionary containing benchmark results.
    """

    torch.manual_seed(42)
    np.random.seed(42)

    a, b = generate_test_data(M, N, K, data_type, device)

    # --- Triton Benchmark ---
    triton_times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)  # Use events for more accurate CUDA timing
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        c_triton = triton_matmul_optimized(a, b)
        end_event.record()
        torch.cuda.synchronize()  # Wait for the GPU to finish
        triton_time_ms = start_event.elapsed_time(end_event)  # Time in milliseconds
        triton_times.append(triton_time_ms / 1000.0)  # Convert to seconds
    triton_median_time = np.median(triton_times)

    # --- PyTorch Benchmark ---
    torch_times = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)  # Use events for more accurate CUDA timing
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        c_torch = torch.matmul(a, b)
        end_event.record()
        torch.cuda.synchronize()  # Wait for the GPU to finish
        torch_time_ms = start_event.elapsed_time(end_event)  # Time in milliseconds
        torch_times.append(torch_time_ms / 1000.0)  # Convert to seconds
    torch_median_time = np.median(torch_times)

    # --- Performance Metrics ---
    flops = 2 * M * N * K  # Standard FLOPS calculation for matrix multiplication
    bytes_accessed = (M * K + K * N + M * N) * torch.finfo(data_type).dtype.itemsize  # Estimate of memory accessed (read + write)

    triton_gflops = flops / triton_median_time / 1e9
    torch_gflops = flops / torch_median_time / 1e9

    triton_gbps = bytes_accessed / triton_median_time / 1e9
    torch_gbps = bytes_accessed / torch_median_time / 1e9

     # Ensure neither time is zero before calculating speedup.  Handle potential division by zero.
    speedup = torch_median_time / triton_median_time if triton_median_time > 0 else np.inf

    return {
        "M": M, "N": N, "K": K,
        "dtype": str(data_type),
        "device": device,
        "Triton Time (s)": triton_median_time,
        "PyTorch Time (s)": torch_median_time,
        "Triton GFLOPS": triton_gflops,
        "PyTorch GFLOPS": torch_gflops,
        "Triton GB/s": triton_gbps,
        "PyTorch GB/s": torch_gbps,
        "Speedup (PyTorch/Triton)": speedup,
        "Block Size M": matmul_kernel_optimized.cache[M, N, K]['BLOCK_SIZE_M'],  # Capture autotuned block sizes
        "Block Size N": matmul_kernel_optimized.cache[M, N, K]['BLOCK_SIZE_N'],
        "Block Size K": matmul_kernel_optimized.cache[M, N, K]['BLOCK_SIZE_K'],
    }

def main():
    """Main function to run the benchmarks and display results."""
    benchmark_cases = [
        {"M": 256, "N": 256, "K": 256, "dtype": torch.float32, "device": "cuda"},
        {"M": 512, "N": 512, "K": 512, "dtype": torch.float32, "device": "cuda"},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": torch.float32, "device": "cuda"},
        {"M": 2048, "N": 2048, "K": 2048, "dtype": torch.float32, "device": "cuda"},
        {"M": 4096, "N": 4096, "K": 4096, "dtype": torch.float32, "device": "cuda"},
        {"M": 256, "N": 256, "K": 256, "dtype": torch.float16, "device": "cuda"},
        {"M": 512, "N": 512, "K": 512, "dtype": torch.float16, "device": "cuda"},
        {"M": 1024, "N": 1024, "K": 1024, "dtype": torch.float16, "device": "cuda"},
        {"M": 2048, "N": 2048, "K": 2048, "dtype": torch.float16, "device": "cuda"}, # Add more float16 cases
        {"M": 4096, "N": 4096, "K": 4096, "dtype": torch.float16, "device": "cuda"}, # Add more float16 cases

        {"M": 1024, "N": 512, "K": 256, "dtype": torch.float32, "device": "cuda"}, # Rectangular matrices
        {"M": 512, "N": 1024, "K": 256, "dtype": torch.float32, "device": "cuda"}, # Rectangular matrices
        {"M": 256, "N": 1024, "K": 512, "dtype": torch.float32, "device": "cuda"}, # Rectangular matrices
    ]

    results = []
    for case in benchmark_cases:
        result = benchmark_matmul(case["M"], case["N"], case["K"], case["dtype"], case["device"])
        results.append(result)

    # --- Display Results in a Table ---
    df = pd.DataFrame(results)
    print(df)

    # --- Analyze Results ---
    print("\n--- Analysis of Results ---")
    for index, row in df.iterrows():
        speedup = row["Speedup (PyTorch/Triton)"]
        M, N, K = row["M"], row["N"], row["K"]
        dtype = row["dtype"]
        device = row["device"]

        if speedup > 1.0:
            print(f"✅ Triton is {speedup:.2f}x faster than PyTorch for M={M}, N={N}, K={K}, dtype={dtype} on {device}.")
        elif speedup < 1.0:
            print(f"⚠️ Triton is {1/speedup:.2f}x slower than PyTorch for M={M}, N={N}, K={K}, dtype={dtype} on {device}.")
        else:
            print(f"⚖️ Triton and PyTorch have approximately the same performance for M={M}, N={N}, K={K}, dtype={dtype} on {device}.")

    # --- Visualization ---
    sns.set_theme(style="darkgrid")

    # 1. Speedup vs. Matrix Size
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df.index, y="Speedup (PyTorch/Triton)", data=df, marker='o')
    plt.xticks(df.index, [f"{row['M']}x{row['N']}x{row['K']}" for _, row in df.iterrows()], rotation=45) # Label with M x N x K
    plt.xlabel("Matrix Size (M x N x K)")
    plt.ylabel("Speedup (PyTorch/Triton)")
    plt.title("Triton vs. PyTorch Speedup by Matrix Size")
    plt.tight_layout()
    plt.savefig("speedup_vs_matrix_size.png") # Save the figure
    plt.show()

    # 2. GFLOPS Comparison (Bar Chart)
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(df))
    plt.bar(x - width/2, df["Triton GFLOPS"], width, label="Triton")
    plt.bar(x + width/2, df["PyTorch GFLOPS"], width, label="PyTorch")
    plt.xticks(x, [f"{row['M']}x{row['N']}x{row['K']}" for _, row in df.iterrows()], rotation=45)
    plt.xlabel("Matrix Size (M x N x K)")
    plt.ylabel("GFLOPS")
    plt.title("GFLOPS Comparison: Triton vs. PyTorch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gflops_comparison.png")
    plt.show()

    # 3. Block Size Heatmap
    block_size_data = df[["M", "N", "K", "Block Size M", "Block Size N", "Block Size K"]]
    block_size_data = block_size_data.melt(id_vars=["M", "N", "K"], var_name="Block Size", value_name="Value")

    plt.figure(figsize=(14, 8))
    sns.heatmap(block_size_data.pivot_table(index=["M", "N", "K"], columns="Block Size", values="Value"),
                annot=True, cmap="viridis", fmt=".0f")
    plt.title("Autotuned Block Sizes for Different Matrix Dimensions")
    plt.tight_layout()
    plt.savefig("autotuned_block_sizes.png")
    plt.show()

if __name__ == "__main__":
    main()
