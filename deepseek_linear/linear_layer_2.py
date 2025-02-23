import torch
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def linear_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, M, K,
    stride_xn, stride_xk,
    stride_wm, stride_wk,
    stride_bm,
    stride_yn, stride_ym,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + offs_n[:, None] * stride_xn + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K - k), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    b_ptrs = b_ptr + offs_m
    b = tl.load(b_ptrs, mask=offs_m < M, other=0.0)
    acc += b[None, :]

    y_ptrs = y_ptr + offs_n[:, None] * stride_yn + offs_m[None, :] * stride_ym
    tl.store(y_ptrs, acc, mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))

@triton.jit
def linear_backward_dx_kernel(
    dy_ptr, w_ptr, dx_ptr,
    N, K, M,
    stride_dyn, stride_dym,
    stride_wm, stride_wk,
    stride_dxn, stride_dxk,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    dy_ptrs = dy_ptr + offs_n[:, None] * stride_dyn + offs_m[None, :] * stride_dym
    w_ptrs = w_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    for m in range(0, M, BLOCK_SIZE_M):
        dy = tl.load(dy_ptrs, mask=(offs_n[:, None] < N) & (offs_m[None, :] < M - m), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(dy, w, allow_tf32=True)
        dy_ptrs += BLOCK_SIZE_M * stride_dym
        w_ptrs += BLOCK_SIZE_M * stride_wm

    dx_ptrs = dx_ptr + offs_n[:, None] * stride_dxn + offs_k[None, :] * stride_dxk
    tl.store(dx_ptrs, acc, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))

@triton.jit
def linear_backward_dw_kernel(
    dy_ptr, x_ptr, dw_ptr,
    M, K, N,
    stride_dym, stride_dyn,
    stride_xn, stride_xk,
    stride_dwm, stride_dwk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    dy_ptrs = dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn
    x_ptrs = x_ptr + offs_n[:, None] * stride_xn + offs_k[None, :] * stride_xk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, N, BLOCK_SIZE_N):
        dy = tl.load(dy_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N - n), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(dy, x, allow_tf32=True)
        dy_ptrs += BLOCK_SIZE_N * stride_dyn
        x_ptrs += BLOCK_SIZE_N * stride_xn

    dw_ptrs = dw_ptr + offs_m[:, None] * stride_dwm + offs_k[None, :] * stride_dwk
    tl.store(dw_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))

@triton.jit
def linear_backward_db_kernel(
    dy_ptr, db_ptr,
    M, N,
    stride_dyn, stride_dym,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    dy_ptrs = dy_ptr + offs_n * stride_dyn + offs_m[:, None] * stride_dym

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE_N):
        dy = tl.load(dy_ptrs, mask=(offs_n < N - n) & (offs_m[:, None] < M), other=0.0)
        acc += tl.sum(dy, axis=0)
        dy_ptrs += BLOCK_SIZE_N * stride_dyn

    tl.store(db_ptr + offs_m, acc, mask=offs_m < M)

class TritonLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight, bias)
        N, K = x.shape
        M, _ = weight.shape
        y = torch.empty((N, M), device=x.device, dtype=x.dtype)
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        linear_forward_kernel[grid](
            x, weight, bias, y,
            N, M, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            bias.stride(0),
            y.stride(0), y.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        N, K = x.shape
        M = weight.shape[0]

        dx = torch.empty_like(x)
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 64
        BLOCK_SIZE_M = 32
        grid_dx = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(K, BLOCK_SIZE_K))
        linear_backward_dx_kernel[grid_dx](
            dy, weight, dx,
            N, K, M,
            dy.stride(0), dy.stride(1),
            weight.stride(0), weight.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_M=BLOCK_SIZE_M,
        )

        dW = torch.empty_like(weight)
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_K = 64
        BLOCK_SIZE_N = 32
        grid_dw = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K))
        linear_backward_dw_kernel[grid_dw](
            dy, x, dW,
            M, K, N,
            dy.stride(1), dy.stride(0),
            x.stride(0), x.stride(1),
            dW.stride(0), dW.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        db = torch.empty_like(bias)
        BLOCK_SIZE_M = 512
        grid_db = (triton.cdiv(M, BLOCK_SIZE_M),)
        linear_backward_db_kernel[grid_db](
            dy, db,
            M, N,
            dy.stride(0), dy.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=256,
        )

        return dx, dW, db

def test_forward():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    weight = torch.randn(512, 256, device='cuda', dtype=torch.float32)
    bias = torch.randn(512, device='cuda', dtype=torch.float32)

    y_triton = TritonLinear.apply(x, weight, bias)
    y_torch = torch.nn.functional.linear(x, weight, bias)

    assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2), "Forward pass failed"

def test_backward():
    torch.manual_seed(0)
    x = torch.randn(128, 256, device='cuda', dtype=torch.float32, requires_grad=True)
    weight = torch.randn(512, 256, device='cuda', dtype=torch.float32, requires_grad=True)
    bias = torch.randn(512, device='cuda', dtype=torch.float32, requires_grad=True)
    dy = torch.randn(128, 512, device='cuda')

    # Triton backward
    y_triton = TritonLinear.apply(x, weight, bias)
    y_triton.backward(dy)
    dx_triton, dw_triton, db_triton = x.grad, weight.grad, bias.grad

    # PyTorch backward
    x.grad = weight.grad = bias.grad = None
    y_torch = torch.nn.functional.linear(x, weight, bias)
    y_torch.backward(dy)
    dx_torch, dw_torch, db_torch = x.grad, weight.grad, bias.grad

    assert torch.allclose(dx_triton, dx_torch, atol=1e-2, rtol=1e-2), "dx incorrect"
    assert torch.allclose(dw_triton, dw_torch, atol=1e-2, rtol=1e-2), "dw incorrect"
    assert torch.allclose(db_triton, db_torch, atol=1e-2, rtol=1e-2), "db incorrect"

def benchmark():
    import time
    batch_size, in_features, out_features = 4096, 4096, 4096
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float32)
    weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_features, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = TritonLinear.apply(x, weight, bias)

    # Triton forward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_triton = TritonLinear.apply(x, weight, bias)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100

    # Torch forward
    torch_layer = torch.nn.Linear(in_features, out_features).cuda()
    torch_layer.weight.data = weight.data
    torch_layer.bias.data = bias.data
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_torch = torch_layer(x)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100

    print(f"Forward: Triton {triton_time:.6f}s vs PyTorch {torch_time:.6f}s")

    # Backward benchmark
    dy = torch.randn_like(y_triton)

    # Warmup
    for _ in range(10):
        y_triton.backward(dy, retain_graph=True)

    # Triton backward
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_triton.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    triton_bwd_time = (time.time() - start) / 100

    # Torch backward
    y_torch = torch_layer(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y_torch.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    torch_bwd_time = (time.time() - start) / 100

    print(f"Backward: Triton {triton_bwd_time:.6f}s vs PyTorch {torch_bwd_time:.6f}s")

if __name__ == "__main__":
    test_forward()
    test_backward()
    print("All tests passed!")
    benchmark()
