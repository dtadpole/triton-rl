import time
import torch
import triton
import triton.language as tl

@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, M, N,
    stride_input_b, stride_input_m,
    stride_weight_n, stride_weight_m,
    stride_output_b, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_b = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_b = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, M, BLOCK_K):
        # Load input block (BLOCK_M, BLOCK_K)
        input_ptrs = input_ptr + offs_b[:, None] * stride_input_b + (k + offs_k[None, :]) * stride_input_m
        input_block = tl.load(input_ptrs, mask=(offs_b[:, None] < B) & ((k + offs_k[None, :]) < M), other=0.0)
        # Load weight block (BLOCK_N, BLOCK_K), transpose to (BLOCK_K, BLOCK_N)
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_weight_n + (k + offs_k[None, :]) * stride_weight_m
        weight_block = tl.trans(tl.load(weight_ptrs, mask=(offs_n[:, None] < N) & ((k + offs_k[None, :]) < M), other=0.0))
        # Matrix multiplication
        accumulator += tl.dot(input_block, weight_block, allow_tf32=True)

    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias_block = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator += bias_block[None, :]

    output_ptrs = output_ptr + offs_b[:, None] * stride_output_b + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, accumulator, mask=(offs_b[:, None] < B) & (offs_n[None, :] < N))


@triton.jit
def linear_backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, N, M,
    stride_dout_b, stride_dout_n,
    stride_weight_n, stride_weight_m,
    stride_din_b, stride_din_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(B, BLOCK_M)
    num_pid_m = tl.cdiv(M, BLOCK_N)
    pid_b = pid // num_pid_m
    pid_m = pid % num_pid_m

    offs_b = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_k[None, :] * stride_dout_n
        weight_ptrs = weight_ptr + offs_k[:, None] * stride_weight_n + offs_m[None, :] * stride_weight_m
        d_output_block = tl.load(d_output_ptrs, mask=(offs_b[:, None] < B) & (offs_k[None, :] < N), other=0.0)
        weight_block = tl.load(weight_ptrs, mask=(offs_k[:, None] < N) & (offs_m[None, :] < M), other=0.0)
        accumulator += tl.dot(d_output_block, weight_block, allow_tf32=True)

    d_input_ptrs = d_input_ptr + offs_b[:, None] * stride_din_b + offs_m[None, :] * stride_din_m
    tl.store(d_input_ptrs, accumulator, mask=(offs_b[:, None] < B) & (offs_m[None, :] < M))


@triton.jit
def linear_backward_weight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, M, N,
    stride_input_b, stride_input_m,
    stride_dout_b, stride_dout_n,
    stride_dw_n, stride_dw_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_M)
    num_pid_m = tl.cdiv(M, BLOCK_N)
    pid_n = pid // num_pid_m
    pid_m = pid % num_pid_m

    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, B, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        d_output_ptrs = d_output_ptr + offs_k[:, None] * stride_dout_b + offs_n[None, :] * stride_dout_n
        input_ptrs = input_ptr + offs_k[:, None] * stride_input_b + offs_m[None, :] * stride_input_m
        d_output_block = tl.load(d_output_ptrs, mask=(offs_k[:, None] < B) & (offs_n[None, :] < N), other=0.0)
        input_block = tl.load(input_ptrs, mask=(offs_k[:, None] < B) & (offs_m[None, :] < M), other=0.0)
        accumulator += tl.dot(d_output_block, input_block, allow_tf32=True)

    dw_ptrs = d_weight_ptr + offs_n[:, None] * stride_dw_n + offs_m[None, :] * stride_dw_m
    tl.store(dw_ptrs, accumulator, mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))

@triton.jit
def linear_backward_bias_kernel(
    d_output_ptr, d_bias_ptr, B, N,
    stride_dout_b, stride_dout_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for b in range(0, B, BLOCK_SIZE):
        offs_b = b + tl.arange(0, BLOCK_SIZE)
        d_output_ptrs = d_output_ptr + offs_b[:, None] * stride_dout_b + offs_n[None, :] * stride_dout_n
        d_output_block = tl.load(d_output_ptrs, mask=(offs_b[:, None] < B) & mask, other=0.0)
        accumulator += tl.sum(d_output_block, axis=0)

    d_bias_ptrs = d_bias_ptr + offs_n
    tl.store(d_bias_ptrs, accumulator, mask=mask)


def linear_triton_forward(input, weight, bias):
    B, M = input.shape
    N, _ = weight.shape
    output = torch.empty((B, N), device=input.device, dtype=input.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
    linear_forward_kernel[grid](
        input, weight, bias, output, B, M, N,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return output

def linear_triton_backward_input(d_output, weight):
    B, N = d_output.shape
    M = weight.shape[1]
    d_input = torch.empty((B, M), device=d_output.device, dtype=d_output.dtype)
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']) * triton.cdiv(M, meta['BLOCK_N']), )
    linear_backward_input_kernel[grid](
        d_output, weight, d_input, B, N, M,
        d_output.stride(0), d_output.stride(1),
        weight.stride(0), weight.stride(1),
        d_input.stride(0), d_input.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return d_input

def linear_triton_backward_weight(input, d_output):
    B, M = input.shape
    N = d_output.shape[1]
    d_weight = torch.empty((N, M), device=input.device, dtype=input.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']) * triton.cdiv(M, meta['BLOCK_N']), )
    linear_backward_weight_kernel[grid](
        input, d_output, d_weight, B, M, N,
        input.stride(0), input.stride(1),
        d_output.stride(0), d_output.stride(1),
        d_weight.stride(0), d_weight.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
    )
    return d_weight

def linear_triton_backward_bias(d_output):
    B, N = d_output.shape
    d_bias = torch.empty(N, device=d_output.device, dtype=d_output.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    linear_backward_bias_kernel[grid](
        d_output, d_bias, B, N,
        d_output.stride(0), d_output.stride(1),
        BLOCK_SIZE=256
    )
    return d_bias

class LinearTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = linear_triton_forward(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None


import pytest
import torch.nn as nn

@pytest.mark.parametrize("B,M,N,dtype", [
    (4, 784, 512, torch.float32),
    (128, 1024, 256, torch.float32),
    (8, 512, 1024, torch.float16)
])
def test_forward(B, M, N, dtype):
    torch.manual_seed(0)
    # Create input data
    x = torch.randn((B, M), device='cuda', dtype=dtype)
    # Initialize both layers with same weights
    layer_triton = LinearTritonFunction
    layer_pytorch = nn.Linear(M, N).cuda().to(dtype)
    with torch.no_grad():
        layer_pytorch.weight.copy_(layer_triton.weight)
        layer_pytorch.bias.copy_(layer_triton.bias)

    # Run both implementations
    out_triton = layer_triton(x)
    out_pytorch = layer_pytorch(x)

    # Compare results
    assert torch.allclose(out_triton, out_pytorch, rtol=1e-2, atol=1e-3)

@pytest.mark.parametrize("B,M,N", [(128, 512, 256)])
def test_backward_input(B, M, N):
    torch.manual_seed(0)
    x = torch.randn((B, M), device='cuda', requires_grad=True)
    grad_output = torch.randn((B, N), device='cuda')

    # Triton backward
    x_triton = x.clone().detach().requires_grad_(False)
    d_input_triton = linear_triton_backward_input(grad_output, weight)

    # PyTorch backward
    x_pytorch = x.clone().detach().requires_grad_(True)
    out_pytorch = x_pytorch @ weight.t() + bias
    out_pytorch.backward(grad_output)

    assert torch.allclose(d_input_triton, x_pytorch.grad, rtol=1e-3, atol=1e-4)

@pytest.mark.parametrize("B,M,N", [(128, 512, 256)])
def test_backward_weight(B, M, N):
    torch.manual_seed(0)
    x = torch.randn((B, M), device='cuda')
    grad_output = torch.randn((B, N), device='cuda')

    # Triton backward
    d_weight_triton = linear_triton_backward_weight(x, grad_output)

    # PyTorch backward
    weight_pytorch = weight.clone().detach().requires_grad_(True)
    out_pytorch = x @ weight_pytorch.t() + bias
    out_pytorch.backward(grad_output)

    assert torch.allclose(d_weight_triton, weight_pytorch.grad, rtol=1e-3, atol=1e-4)

@pytest.mark.parametrize("B,N", [(128, 256)])
def test_backward_bias(B, N):
    torch.manual_seed(0)
    grad_output = torch.randn((B, N), device='cuda')

    # Triton backward
    d_bias_triton = linear_triton_backward_bias(grad_output)

    # PyTorch backward
    bias_pytorch = bias.clone().detach().requires_grad_(True)
    out_pytorch = torch.zeros(B, device='cuda') + bias_pytorch  # Dummy operation
    out_pytorch.backward(grad_output.sum(dim=0))  # PyTorch sums gradients

    assert torch.allclose(d_bias_triton, bias_pytorch.grad, rtol=1e-3, atol=1e-4)

def test_autograd():
    # Verify integration with PyTorch autograd engine
    input = torch.randn(16, 512, device='cuda', dtype=torch.double, requires_grad=True)
    weight = torch.randn(256, 512, device='cuda', dtype=torch.double, requires_grad=True)
    bias = torch.randn(256, device='cuda', dtype=torch.double, requires_grad=True)

    test = torch.autograd.gradcheck(
        LinearTritonFunction.apply,
        (input, weight, bias),
        eps=1e-4,
        atol=1e-3,
        rtol=1e-3
    )
    assert test

def benchmark_linear():
    sizes = [
        (1, 1024, 4096),
        (128, 512, 1024),
        (256, 1024, 2048)
    ]

    for B, M, N in sizes:
        # Create inputs
        x = torch.randn(B, M, device='cuda')
        weight = torch.randn(N, M, device='cuda')
        bias = torch.randn(N, device='cuda')

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            linear_triton_forward(x, weight, bias)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100

        # Benchmark PyTorch
        layer = nn.Linear(M, N).cuda()
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            layer(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100

        print(f"Batch {B}, Input {M}, Output {N}")
        print(f"Triton: {triton_time*1e6:.2f}μs")
        print(f"PyTorch: {pytorch_time*1e6:.2f}μs")
        print(f"Speedup: {pytorch_time/triton_time:.2f}x\n")

if __name__ == "__main__":
    benchmark_linear()
