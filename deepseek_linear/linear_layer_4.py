import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
    ],
    key=['B', 'M', 'N'],
)
@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, M, N,
    stride_input_b, stride_input_m,
    stride_weight_n, stride_weight_m,
    stride_output_b, stride_output_n,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_M)

    input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + offs_k[None, :] * stride_input_m
    weight_ptrs = weight_ptr + offs_n[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_m

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        k_remaining = M - k * BLOCK_SIZE_M
        input_mask = offs_k[None, :] < k_remaining
        weight_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)

        input = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        accumulator += tl.dot(input, tl.trans(weight))

        input_ptrs += BLOCK_SIZE_M * stride_input_m
        weight_ptrs += BLOCK_SIZE_M * stride_weight_m

    offs_output_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_output_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + offs_output_b[:, None] * stride_output_b + offs_output_n[None, :] * stride_output_n
    output_mask = (offs_output_b[:, None] < B) & (offs_output_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
    ],
    key=['B', 'M', 'N'],
)
@triton.jit
def linear_backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, N, M,
    stride_dout_b, stride_dout_n,
    stride_weight_n, stride_weight_m,
    stride_din_b, stride_din_m,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_k = k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        dout_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_k[None, :] * stride_dout_n
        weight_ptrs = weight_ptr + offs_k[:, None] * stride_weight_n + offs_m[None, :] * stride_weight_m

        dout_mask = offs_k[None, :] < (N - k * BLOCK_SIZE_N)
        weight_mask = (offs_k[:, None] < N) & (offs_m[None, :] < M)

        dout = tl.load(dout_ptrs, mask=dout_mask, other=0.0)
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        accumulator += tl.dot(dout, weight)

    d_input_ptrs = d_input_ptr + offs_b[:, None] * stride_din_b + offs_m[None, :] * stride_din_m
    din_mask = (offs_b[:, None] < B) & (offs_m[None, :] < M)
    tl.store(d_input_ptrs, accumulator, mask=din_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE_B': 8}, num_warps=4),
    ],
    key=['B', 'M', 'N'],
)
@triton.jit
def linear_backward_weight_kernel(
    d_output_ptr, input_ptr, d_weight_ptr,
    B, N, M,
    stride_dout_b, stride_dout_n,
    stride_input_b, stride_input_m,
    stride_dw_n, stride_dw_m,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_B: tl.constexpr,
    GROUP_SIZE_B: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
        offs_k = k * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        dout_ptrs = d_output_ptr + offs_k[:, None] * stride_dout_b + offs_n[None, :] * stride_dout_n
        input_ptrs = input_ptr + offs_k[:, None] * stride_input_b + offs_m[None, :] * stride_input_m

        dout_mask = (offs_k[:, None] < B) & (offs_n[None, :] < N)
        input_mask = (offs_k[:, None] < B) & (offs_m[None, :] < M)

        dout = tl.load(dout_ptrs, mask=dout_mask, other=0.0)
        input = tl.load(input_ptrs, mask=input_mask, other=0.0)

        accumulator += tl.dot(dout.T, input)

    dw_ptrs = d_weight_ptr + offs_n[:, None] * stride_dw_n + offs_m[None, :] * stride_dw_m
    dw_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(dw_ptrs, accumulator, mask=dw_mask)

class TritonLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        B, M = input.shape
        N, _ = weight.shape
        output = torch.empty((B, N), device=input.device, dtype=input.dtype)  # (B, N)

        grid = lambda META: (triton.cdiv(B, META['BLOCK_SIZE_B']), triton.cdiv(N, META['BLOCK_SIZE_N']))
        linear_forward_kernel[grid](
            input, weight, output,
            B, M, N,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        B, M = input.shape
        N, _ = weight.shape

        grad_input = torch.empty_like(input)  # (B, M)
        grad_weight = torch.empty_like(weight)  # (N, M)

        grid_din = lambda META: (triton.cdiv(B, META['BLOCK_SIZE_B']), triton.cdiv(N, META['BLOCK_SIZE_N']))
        linear_backward_input_kernel[grid_din](
            grad_output, weight, grad_input,
            B, N, M,
            grad_output.stride(0), grad_output.stride(1),
            weight.stride(0), weight.stride(1),
            grad_input.stride(0), grad_input.stride(1),
        )

        grid_dw = lambda META: (triton.cdiv(B, META['BLOCK_SIZE_B']), triton.cdiv(N, META['BLOCK_SIZE_N']))
        linear_backward_weight_kernel[grid_dw](
            grad_output, input, grad_weight,
            B, N, M,
            grad_output.stride(0), grad_output.stride(1),
            input.stride(0), input.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
        )
        return grad_input, grad_weight

# Test cases
def test_forward():
    torch.manual_seed(0)
    B, M, N = 128, 784, 256
    input = torch.randn(B, M, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, M, device='cuda', dtype=torch.float16)

    output_triton = TritonLinear.apply(input, weight)
    output_torch = F.linear(input, weight)

    if torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2):
        print("✅ Forward pass passed!")
    else:
        print("❌ Forward pass failed!")

def test_backward():
    torch.manual_seed(0)
    B, M, N = 128, 784, 256
    input = torch.randn(B, M, device='cuda', dtype=torch.float16, requires_grad=True)
    weight = torch.randn(N, M, device='cuda', dtype=torch.float16, requires_grad=True)
    grad_output = torch.randn(B, N, device='cuda', dtype=torch.float16)

    output_triton = TritonLinear.apply(input, weight)
    output_triton.backward(grad_output, retain_graph=True)
    grad_input_triton, grad_weight_triton = input.grad.clone(), weight.grad.clone()

    input.grad = None
    weight.grad = None
    output_torch = F.linear(input, weight)
    output_torch.backward(grad_output)

    if torch.allclose(grad_input_triton, input.grad, atol=1e-2, rtol=1e-2):
        print("✅ Backward input grad passed!")
    else:
        print("❌ Backward input grad failed!")
    if torch.allclose(grad_weight_triton, weight.grad, atol=1e-2, rtol=1e-2):
        print("✅ Backward weight grad passed!")
    else:
        print("❌ Backward weight grad failed!")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],  # Dimension to vary
        x_vals=[256 * i for i in range(1, 5)],  # Different values for M
        line_arg='provider',  # Argument name for different implementations
        line_vals=['triton', 'torch'],  # Labels for the lines
        line_names=['Triton', 'Torch'],  # Line labels
        styles=[('blue', '-'), ('green', '-')],  # Line styles
        ylabel="GB/s",  # Label for Y axis
        plot_name="linear-forward",  # Name for the plot
        args={'B': 128, 'N': 256},  # Fixed arguments
    )
)
def benchmark(B, N, M, provider):
    input = torch.randn(B, M, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, M, device='cuda', dtype=torch.float16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: TritonLinear.apply(input, weight))
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: F.linear(input, weight))

    gbps = lambda ms: 2 * B * M * N * 1e-9 / (ms * 1e-3)
    return gbps(ms)

# Run tests and benchmark
if __name__ == "__main__":
    test_forward()
    test_backward()
    benchmark.run(print_data=True, show_plots=True)
