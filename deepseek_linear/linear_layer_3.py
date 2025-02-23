import torch
import triton
import triton.language as tl
import time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_forward_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_curr = min(BLOCK_SIZE_K, k_remaining)
        x = tl.load(x_ptrs, mask=(offs_k[None, :] < k_curr), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[None, :] < k_curr), other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_backward_dx_kernel(
    dy_ptr, w_ptr, dx_ptr,
    M, N, K,
    stride_dym, stride_dyn,
    stride_wk, stride_wn,
    stride_dxm, stride_dxk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dy_ptrs = dy_ptr + (offs_m[:, None] * stride_dym + offs_k[None, :] * stride_dyn)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_curr = min(BLOCK_SIZE_K, k_remaining)
        dy = tl.load(dy_ptrs, mask=(offs_k[None, :] < k_curr), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_curr), other=0.0)
        acc += tl.dot(dy, w, allow_tf32=True)
        dy_ptrs += BLOCK_SIZE_K * stride_dyn
        w_ptrs += BLOCK_SIZE_K * stride_wk

    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxk)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(dx_ptrs, acc, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_backward_dw_kernel(
    dy_ptr, x_ptr, dw_ptr,
    M, N, K,
    stride_dy_out, stride_dy_n,
    stride_xn, stride_xk,
    stride_dw_out, stride_dw_in,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dy_ptrs = dy_ptr + (offs_m[:, None] * stride_dy_out + offs_k[None, :] * stride_dy_n)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_xn + offs_n[None, :] * stride_xk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_curr = min(BLOCK_SIZE_K, k_remaining)
        a = tl.load(dy_ptrs, mask=(offs_k[None, :] < k_curr), other=0.0)
        b = tl.load(x_ptrs, mask=(offs_k[:, None] < k_curr), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        dy_ptrs += BLOCK_SIZE_K * stride_dy_n
        x_ptrs += BLOCK_SIZE_K * stride_xn

    dw_ptrs = dw_ptr + (offs_m[:, None] * stride_dw_out + offs_n[None, :] * stride_dw_in)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(dw_ptrs, acc, mask=mask)

class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        n, in_features = x.shape
        out_features = weight.shape[0]
        y = torch.empty((n, out_features), device=x.device, dtype=x.dtype)
        M, N, K = n, out_features, in_features
        grid = lambda opt: (
            triton.cdiv(M, opt['BLOCK_SIZE_M']) * triton.cdiv(N, opt['BLOCK_SIZE_N']),
        )
        linear_forward_kernel[grid](
            x, weight, y,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            y.stride(0), y.stride(1),
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        n, in_features = x.shape
        out_features = weight.shape[0] # (out_features, in_features)

        dx = torch.empty_like(x) # (n, in_features)
        M_dx, N_dx, K_dx = n, in_features, out_features
        grid_dx = lambda opt: (
            triton.cdiv(M_dx, opt['BLOCK_SIZE_M']) * triton.cdiv(N_dx, opt['BLOCK_SIZE_N']),
        )
        linear_backward_dx_kernel[grid_dx](
            dy, weight, dx,
            M_dx, N_dx, K_dx,
            dy.stride(0), dy.stride(1), # (n, out_features)
            weight.stride(0), weight.stride(1), # (out_features, in_features)
            dx.stride(0), dx.stride(1), # (n, in_features)
        )

        dw = torch.empty_like(weight)
        M_dw, N_dw, K_dw = out_features, in_features, n
        stride_dy_out = 1  # Since dy is (n, out), dy.T's row stride is 1
        stride_dy_n = dy.stride(0)  # dy.T's column stride is original dy's row stride
        grid_dw = lambda opt: (
            triton.cdiv(M_dw, opt['BLOCK_SIZE_M']) * triton.cdiv(N_dw, opt['BLOCK_SIZE_N']),
        )
        linear_backward_dw_kernel[grid_dw](
            dy, x, dw,
            M_dw, N_dw, K_dw,
            stride_dy_out, stride_dy_n, # (out_features, n)
            x.stride(0), x.stride(1), # (n, in_features)
            dw.stride(0), dw.stride(1), # (out_features, in_features)
        )

        return dx, dw

class LinearTriton(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self, param_init=None):
        if param_init is None:
            self.weight = torch.randn(self.out_features, self.in_features, device='cuda')
            self.weight = torch.nn.functional.normalize(self.weight, dim=-1).requires_grad_()
        else:
            self.weight = param_init.clone().detach().requires_grad_()

    def forward(self, x):
        return TritonLinearFunction.apply(x, self.weight)

    def backward(self, dy):
        return self.TritonLinearFunction.backward(dy)


def test_forward():
    torch.manual_seed(0)
    in_features = 512
    out_features = 2048
    batch_size = 768
    x = torch.randn(batch_size, in_features, device='cuda')
    x = torch.nn.functional.normalize(x, dim=1)

    linear_torch = torch.nn.Linear(in_features, out_features, bias=False, device='cuda')
    linear_triton = LinearTriton(in_features, out_features)
    linear_triton.reset_parameters(linear_torch.weight)

    # weight = torch.randn(out_features, in_features, device='cuda')
    # weight = torch.nn.functional.normalize(weight, dim=1)
    # y_triton = TritonLinearFunction.apply(x, weight)
    # y_pytorch = torch.nn.functional.linear(x, weight, bias=None)
    y_triton = linear_triton(x)
    y_pytorch = linear_torch(x)
    if torch.allclose(y_triton, y_pytorch, rtol=1e-2, atol=1e-2):
        print("✅ Forward pass passed!")
    else:
        print("❌ Forward pass failed!")

def test_backward():
    torch.manual_seed(0)
    in_features = 512
    out_features = 2048
    batch_size = 768
    x = torch.randn(batch_size, in_features, device='cuda', requires_grad=True)
    x = torch.nn.functional.normalize(x, dim=1)

    linear_torch = torch.nn.Linear(in_features, out_features, bias=False, device='cuda')
    linear_triton = LinearTriton(in_features, out_features)
    linear_triton.reset_parameters(linear_torch.weight)
    # weight = torch.randn(out_features, in_features, device='cuda', requires_grad=True)
    # weight = torch.nn.functional.normalize(weight, dim=1)

    dy = torch.randn(batch_size, out_features, device='cuda')
    dy = torch.nn.functional.normalize(dy, dim=1)

    # Triton backward
    x_triton = x.clone().detach().requires_grad_()
    y_triton = linear_triton(x_triton)
    y_triton.backward(dy)
    dx_triton, dw_triton = x_triton.grad, linear_triton.weight.grad

    # PyTorch backward
    x_pytorch = x.clone().detach().requires_grad_()
    y_pytorch = linear_torch(x_pytorch)
    y_pytorch.backward(dy)
    dx_pytorch, dw_pytorch = x_pytorch.grad, linear_torch.weight.grad

    if torch.allclose(dx_triton, dx_pytorch, rtol=1e-2, atol=1e-2):
        print("✅ Backward dx grad passed!")
    else:
        print("❌ Backward dx grad failed!")
    if torch.allclose(dw_pytorch, dw_triton, rtol=1e-2, atol=1e-2):
        print("✅ Backward dw grad passed!")
    else:
        print("❌ Backward dw grad failed!")

def benchmark(device='cuda', forward=True, backward=True):
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    for (M, N, K) in sizes:
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        weight = torch.randn(N, K, device=device, dtype=torch.float16)
        dy = torch.randn(M, N, device=device, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        if forward:
            print(f'Forward pass - {M}x{K} @ {N}x{K}^T -> {M}x{N}')
            triton_time = triton.testing.do_bench(lambda: TritonLinearFunction.apply(x, weight), quantiles=quantiles)
            torch_time = triton.testing.do_bench(lambda: torch.nn.functional.linear(x, weight, bias=None), quantiles=quantiles)
            print(f'Triton: {triton_time[0]:.3f} ms')
            print(f'PyTorch: {torch_time[0]:.3f} ms')
            print(f'Speedup: {torch_time[0]/triton_time[0]:.2f}x')

        if backward:
            print(f'Backward pass - dX: {M}x{N} @ {N}x{K} -> {M}x{K}, dW: {N}x{M} @ {M}x{K} -> {N}x{K}')
            def triton_backward():
                x_triton = x.clone().detach().requires_grad_()
                weight_triton = weight.clone().detach().requires_grad_()
                y_triton = TritonLinearFunction.apply(x_triton, weight_triton)
                y_triton.backward(dy)
                dx_triton, dw_triton = x_triton.grad, weight_triton.grad
            def torch_backward():
                x_torch = x.clone().detach().requires_grad_()
                weight_torch = weight.clone().detach().requires_grad_()
                y = torch.nn.functional.linear(x_torch, weight_torch, bias=None)
                y.backward(dy)
                dx_torch, dw_torch = x_torch.grad, weight_torch.grad
            triton_time = triton.testing.do_bench(triton_backward, quantiles=quantiles)
            torch_time = triton.testing.do_bench(torch_backward, quantiles=quantiles)
            print(f'Triton: {triton_time[0]:.3f} ms')
            print(f'PyTorch: {torch_time[0]:.3f} ms')
            print(f'Speedup: {torch_time[0]/triton_time[0]:.2f}x')

if __name__ == '__main__':
    test_forward()
    test_backward()
    print("Benchmarking...")
    benchmark()
