import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def linear_forward_kernel(
    input_ptr, weight_ptr, output_ptr,
    B, M, N,
    stride_xb, stride_xm,
    stride_wm, stride_wn,
    stride_yb, stride_yn,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, M, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)

        x = tl.load(input_ptr + off_b[:, None] * stride_xb + off_k[None, :] * stride_xm,
                    mask=(off_b[:, None] < B) & (off_k[None, :] < M), other=0.0)
        w = tl.load(weight_ptr + off_k[:, None] * stride_wm + off_n[None, :] * stride_wn,
                    mask=(off_k[:, None] < M) & (off_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)

    tl.store(output_ptr + off_b[:, None] * stride_yb + off_n[None, :] * stride_yn,
             acc, mask=(off_b[:, None] < B) & (off_n[None, :] < N))

def triton_linear_forward(x, w):
    B, M = x.shape
    N, _ = w.shape
    y = torch.empty((B, N), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_B = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(B, BLOCK_SIZE_B), triton.cdiv(N, BLOCK_SIZE_N))

    linear_forward_kernel[grid](
        x, w, y,
        B, M, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return y

@triton.jit
def linear_backward_input_kernel(
    d_output_ptr, weight_ptr, d_input_ptr,
    B, N, M,
    stride_dout_b, stride_dout_n,
    stride_w_n, stride_w_m,
    stride_din_b, stride_din_m,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    off_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    acc = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, N, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load d_output tile: (BLOCK_SIZE_B, BLOCK_SIZE_K)
        dout_tile = tl.load(d_output_ptr + off_b[:, None] * stride_dout_b + off_k[None, :] * stride_dout_n,
                            mask=(off_b[:, None] < B) & (off_k[None, :] < N), other=0.0)
        # Load weight tile: (BLOCK_SIZE_K, BLOCK_SIZE_M)
        w_tile = tl.load(weight_ptr + off_k[:, None] * stride_w_n + off_m[None, :] * stride_w_m,
                         mask=(off_k[:, None] < N) & (off_m[None, :] < M), other=0.0)
        acc += tl.dot(dout_tile, w_tile)

    # Write d_input (B, M)
    tl.store(d_input_ptr + off_b[:, None] * stride_din_b + off_m[None, :] * stride_din_m,
             acc, mask=(off_b[:, None] < B) & (off_m[None, :] < M))


@triton.jit
def linear_backward_weight_kernel(
    input_ptr, d_output_ptr, d_weight_ptr,
    B, M, N,
    stride_x_b, stride_x_m,
    stride_dout_b, stride_dout_n,
    stride_dw_m, stride_dw_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, B, BLOCK_SIZE_K):
        off_k = k + tl.arange(0, BLOCK_SIZE_K)

        # Load input tile: (BLOCK_SIZE_K, BLOCK_SIZE_M)
        x_tile = tl.load(input_ptr + off_k[:, None] * stride_x_b + off_m[None, :] * stride_x_m,
                         mask=(off_k[:, None] < B) & (off_m[None, :] < M), other=0.0)
        # Load d_output tile: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        dout_tile = tl.load(d_output_ptr + off_k[:, None] * stride_dout_b + off_n[None, :] * stride_dout_n,
                            mask=(off_k[:, None] < B) & (off_n[None, :] < N), other=0.0)
        acc += tl.dot(x_tile.T, dout_tile)  # Transpose x_tile for matmul

    # Write d_weight (M, N)
    tl.store(d_weight_ptr + off_m[:, None] * stride_dw_m + off_n[None, :] * stride_dw_n,
             acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))

def test_forward():
    B, M, N = 128, 256, 512
    x = torch.randn(B, M, device='cuda', dtype=torch.float16)
    x = nn.functional.normalize(x, dim=1)
    w = torch.randn(N, M, device='cuda', dtype=torch.float16)
    w = nn.functional.normalize(w, dim=1)

    # Triton forward
    y_triton = triton_linear_forward(x, w)

    # PyTorch forward
    y_pytorch = x @ w.T

    assert torch.allclose(y_triton, y_pytorch, atol=1e-2, rtol=1e-2), "Forward pass failed"
    print("Forward pass test passed!")

def test_backward():
    B, M, N = 128, 256, 512
    x = torch.randn(B, M, device='cuda', dtype=torch.float16, requires_grad=True)
    w = torch.randn(N, M, device='cuda', dtype=torch.float16, requires_grad=True)
    d_output = torch.randn(B, N, device='cuda', dtype=torch.float16)

    # PyTorch backward
    y_pytorch = x @ w.T
    y_pytorch.backward(d_output)
    din_pytorch, dw_pytorch = x.grad, w.grad

    # Triton backward
    din_triton = torch.empty_like(x)
    dw_triton = torch.empty_like(w)

    # Launch input gradient kernel
    grid_din = (triton.cdiv(B, 32), triton.cdiv(M, 64))
    linear_backward_input_kernel[grid_din](
        d_output, w, din_triton,
        B, N, M,
        d_output.stride(0), d_output.stride(1),
        w.stride(0), w.stride(1),
        din_triton.stride(0), din_triton.stride(1),
        BLOCK_SIZE_B=32, BLOCK_SIZE_M=64, BLOCK_SIZE_K=32
    )

    # Launch weight gradient kernel
    grid_dw = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    linear_backward_weight_kernel[grid_dw](
        x, d_output, dw_triton,
        B, M, N,
        x.stride(0), x.stride(1),
        d_output.stride(0), d_output.stride(1),
        dw_triton.stride(0), dw_triton.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
    )

    assert torch.allclose(din_triton, din_pytorch, atol=1e-2, rtol=1e-2), "d_input mismatch"
    assert torch.allclose(dw_triton, dw_pytorch, atol=1e-2, rtol=1e-2), "d_weight mismatch"
    print("Backward pass test passed!")

def benchmark():
    B, M, N = 1024, 2048, 4096
    x = torch.randn(B, M, device='cuda', dtype=torch.float16)
    w = torch.randn(N, M, device='cuda', dtype=torch.float16)

    # Benchmark Triton forward
    def triton_forward():
        return triton_linear_forward(x, w)
    t_triton = triton.testing.do_bench(triton_forward)

    # Benchmark PyTorch forward
    def pytorch_forward():
        return x @ w.T
    t_pytorch = triton.testing.do_bench(pytorch_forward)

    print(f"Forward: Triton {t_triton:.3f} ms vs PyTorch {t_pytorch:.3f} ms")

    # Benchmark backward (similar structure)
    # ...

if __name__ == "__main__":
    test_forward()
    test_backward()
    benchmark()
