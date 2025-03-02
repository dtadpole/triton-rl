import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


# Forward pass implementation for linear layer (y = xA^T)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute y = x @ w.transpose(0, 1)
    """
    # Matrix multiplication: Y[b, n] = X[b, k] @ W[n, k]
    # W is stored as [out_features, in_features]
    # We need to do batched X @ W.T

    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block start indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Offset pointers for the current block
    x_block_ptr = x_ptr + m_start * in_features  # [batch_size, in_features]
    w_block_ptr = w_ptr + n_start * in_features  # [out_features, in_features]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over blocks along the K dimension
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load input block
        k_remaining = min(BLOCK_SIZE_K, in_features - k)
        x_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * in_features + k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        x_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < in_features)
        x_block = tl.load(x_block_ptr + x_offsets, mask=x_mask, other=0.0)  # [BLOCK_SIZE_M, BLOCK_SIZE_K]

        # Load weight block
        w_offsets = tl.arange(0, BLOCK_SIZE_N)[:, None] * in_features + k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        w_mask = (n_start + tl.arange(0, BLOCK_SIZE_N)[:, None] < out_features) & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < in_features)
        w_block = tl.load(w_block_ptr + w_offsets, mask=w_mask, other=0.0)  # [BLOCK_SIZE_N, BLOCK_SIZE_K]

        # Matrix multiplication for this block
        acc += tl.dot(x_block, tl.trans(w_block))  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # Store output
    output_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * out_features + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    output_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < out_features)
    tl.store(output_ptr + output_offsets, acc, mask=output_mask)


# Backward pass implementation for gradients with respect to input (dL/dx = dL/dy @ w)
@triton.jit
def linear_backward_input_kernel(
    # Pointers to matrices
    grad_output_ptr, w_ptr, grad_input_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute dx = dy @ w
    """
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block start indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Offset pointers for the current block
    grad_output_block_ptr = grad_output_ptr + m_start * out_features  # [batch_size, out_features]
    grad_input_block_ptr = grad_input_ptr + m_start * in_features + n_start  # [batch_size, in_features]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over blocks along the K dimension
    for k in range(0, out_features, BLOCK_SIZE_K):
        # Load grad_output block
        k_remaining = min(BLOCK_SIZE_K, out_features - k)
        grad_output_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * out_features + k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        grad_output_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < out_features)
        grad_output_block = tl.load(grad_output_block_ptr + grad_output_offsets, mask=grad_output_mask, other=0.0)  # [BLOCK_SIZE_M, BLOCK_SIZE_K]

        # Load weight block
        w_offsets = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * in_features + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
        w_mask = (k + tl.arange(0, BLOCK_SIZE_K)[:, None] < out_features) & (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < in_features)
        w_block = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # Matrix multiplication for this block
        acc += tl.dot(grad_output_block, w_block)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # Store output
    grad_input_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * in_features + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    grad_input_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < batch_size) & (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < in_features)
    tl.store(grad_input_block_ptr, acc, mask=grad_input_mask)


# Backward pass implementation for gradients with respect to weights (dL/dw = dL/dy.T @ x)
@triton.jit
def linear_backward_weight_kernel(
    # Pointers to matrices
    grad_output_ptr, x_ptr, grad_weight_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute dw = dy.transpose(0, 1) @ x
    """
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Block start indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Offset pointers for the current block
    grad_weight_block_ptr = grad_weight_ptr + m_start * in_features + n_start  # [out_features, in_features]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over blocks along the K dimension (batch dimension)
    for k in range(0, batch_size, BLOCK_SIZE_K):
        k_remaining = min(BLOCK_SIZE_K, batch_size - k)

        # Load grad_output block
        grad_output_offsets = (k + tl.arange(0, k_remaining))[:, None] * out_features + m_start + tl.arange(0, BLOCK_SIZE_M)[None, :]
        grad_output_mask = (k + tl.arange(0, BLOCK_SIZE_K)[:, None] < batch_size) & (m_start + tl.arange(0, BLOCK_SIZE_M)[None, :] < out_features)
        grad_output_block = tl.load(grad_output_ptr + grad_output_offsets, mask=grad_output_mask, other=0.0)  # [BLOCK_SIZE_K, BLOCK_SIZE_M]

        # Load input block
        x_offsets = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * in_features + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
        x_mask = (k + tl.arange(0, BLOCK_SIZE_K)[:, None] < batch_size) & (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < in_features)
        x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # Matrix multiplication for this block
        acc += tl.dot(tl.trans(grad_output_block), x_block)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

    # Store output
    grad_weight_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * in_features + n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    grad_weight_mask = (m_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < out_features) & (n_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < in_features)
    tl.store(grad_weight_block_ptr, acc, mask=grad_weight_mask)


# Wrapper function for forward pass
def linear_forward(x, weight):
    """
    Forward pass for linear layer: y = x @ w.T
    """
    batch_size, in_features = x.shape  # [B, in_features]
    out_features = weight.shape[0]     # [out_features, in_features]

    # Allocate output
    y = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)  # [B, out_features]

    # Block sizes for efficient execution
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid for parallelization
    grid = (
        triton.cdiv(batch_size, BLOCK_SIZE_M),
        triton.cdiv(out_features, BLOCK_SIZE_N)
    )

    # Launch kernel
    linear_forward_kernel[grid](
        x, weight, y,
        batch_size, in_features, out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return y


# Wrapper function for backward pass
def linear_backward(grad_output, x, weight):
    """
    Backward pass for linear layer
    Computes:
    - grad_input = grad_output @ weight
    - grad_weight = grad_output.T @ x
    """
    batch_size, out_features = grad_output.shape  # [B, out_features]
    _, in_features = x.shape                     # [B, in_features]

    # Allocate outputs
    grad_input = torch.empty((batch_size, in_features), device=grad_output.device, dtype=grad_output.dtype)  # [B, in_features]
    grad_weight = torch.empty((out_features, in_features), device=grad_output.device, dtype=grad_output.dtype)  # [out_features, in_features]

    # Block sizes for efficient execution
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32

    # Grid for parallelization - input gradients
    grid_input = (
        triton.cdiv(batch_size, BLOCK_SIZE_M),
        triton.cdiv(in_features, BLOCK_SIZE_N),
    )

    # Grid for parallelization - weight gradients
    grid_weight = (
        triton.cdiv(out_features, BLOCK_SIZE_M),
        triton.cdiv(in_features, BLOCK_SIZE_N),
    )

    # Launch kernels
    linear_backward_input_kernel[grid_input](
        grad_output, weight, grad_input,
        batch_size, in_features, out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    linear_backward_weight_kernel[grid_weight](
        grad_output, x, grad_weight,
        batch_size, in_features, out_features,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return grad_input, grad_weight


# Custom autograd function that uses our Triton kernels
class TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return linear_forward(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_input, grad_weight = linear_backward(grad_output, x, weight)
        return grad_input, grad_weight


# User-friendly module that wraps our implementation
class LinearLayerTriton(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))  # [out_features, in_features]
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return TritonLinear.apply(x, self.weight)


# Test case to verify correctness
def test_linear_layer():
    print("Testing linear layer implementation...")

    # Test configurations
    batch_sizes = [32, 128, 1024]
    in_features_list = [64, 256, 1024]
    out_features_list = [64, 256, 1024]

    device = torch.device('cuda')

    for batch_size in batch_sizes:
        for in_features in in_features_list:
            for out_features in out_features_list:
                print(f"Testing B={batch_size}, in_features={in_features}, out_features={out_features}")

                # Create inputs
                x = torch.randn(batch_size, in_features, device=device, requires_grad=True)  # [B, in_features]
                weight = torch.randn(out_features, in_features, device=device, requires_grad=True)  # [out_features, in_features]

                # Forward pass - PyTorch
                y_pytorch = F.linear(x, weight)  # [B, out_features]

                # Forward pass - Triton
                y_triton = linear_forward(x, weight)  # [B, out_features]

                # Check forward pass
                assert torch.allclose(y_pytorch, y_triton, rtol=1e-2, atol=1e-2), \
                    f"Forward pass outputs don't match: {torch.max(torch.abs(y_pytorch - y_triton))}"

                # Create gradient for backward pass
                grad_output = torch.randn_like(y_pytorch)  # [B, out_features]

                # Backward pass - PyTorch
                y_pytorch.backward(grad_output)
                pytorch_grad_x = x.grad.clone()  # [B, in_features]
                pytorch_grad_weight = weight.grad.clone()  # [out_features, in_features]

                # Reset gradients
                x.grad.zero_()
                weight.grad.zero_()

                # Backward pass - Triton
                grad_input, grad_weight = linear_backward(grad_output, x, weight)

                # Check backward pass
                assert torch.allclose(pytorch_grad_x, grad_input, rtol=1e-2, atol=1e-2), \
                    f"Gradient w.r.t input doesn't match: {torch.max(torch.abs(pytorch_grad_x - grad_input))}"

                assert torch.allclose(pytorch_grad_weight, grad_weight, rtol=1e-2, atol=1e-2), \
                    f"Gradient w.r.t weight doesn't match: {torch.max(torch.abs(pytorch_grad_weight - grad_weight))}"

                print("  ✓ Test passed")

    print("All tests passed!")


# Benchmark performance
def benchmark_linear_layer():
    print("Benchmarking linear layer implementations...")

    # Test configurations
    configs = [
        # (batch_size, in_features, out_features)
        (64, 128, 128),
        (128, 256, 256),
        (256, 512, 512),
        (512, 1024, 1024),
        (1024, 2048, 2048),
    ]

    device = torch.device('cuda')

    # Warm-up GPU
    x = torch.randn(1024, 1024, device=device)
    w = torch.randn(1024, 1024, device=device)
    for _ in range(10):
        y = F.linear(x, w)
        y = linear_forward(x, w)

    # For plotting
    config_names = [f"B={b}, in={i}, out={o}" for b, i, o in configs]
    pytorch_times = []
    triton_times = []
    speedups = []

    for batch_size, in_features, out_features in configs:
        print(f"Benchmarking B={batch_size}, in_features={in_features}, out_features={out_features}")

        # Create inputs
        x = torch.randn(batch_size, in_features, device=device)  # [B, in_features]
        weight = torch.randn(out_features, in_features, device=device)  # [out_features, in_features]

        # Benchmark PyTorch
        torch.cuda.synchronize()
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            y_pytorch = F.linear(x, weight)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / iterations * 1000  # ms

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            y_triton = linear_forward(x, weight)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / iterations * 1000  # ms

        # Calculate speedup
        speedup = pytorch_time / triton_time

        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  Triton:  {triton_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        pytorch_times.append(pytorch_time)
        triton_times.append(triton_time)
        speedups.append(speedup)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    width = 0.35
    x = range(len(configs))

    # Bar chart of times
    ax1.bar([i - width/2 for i in x], pytorch_times, width, label='PyTorch')
    ax1.bar([i + width/2 for i in x], triton_times, width, label='Triton')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.legend()

    # Bar chart of speedups
    ax2.bar(x, speedups)
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Triton Speedup over PyTorch')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_names, rotation=45, ha='right')

    fig.tight_layout()
    plt.savefig('linear_benchmark.png')
    plt.close()


if __name__ == "__main__":
    # Run tests to verify correctness
    test_linear_layer()

    # Run benchmarks to evaluate performance
    benchmark_linear_layer()
