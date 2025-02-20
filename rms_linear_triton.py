import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_linear_kernel(
    # Pointers to matrices
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # Matrix dimensions
    M,  # Number of input rows (batch size)
    N,  # Number of input columns / RMSNorm dimension
    K,  # Number of output columns (linear layer output dimension)
    # Strides
    stride_xm,
    stride_xn,
    stride_wm,
    stride_wn,
    stride_bm,
    stride_ok,
    stride_om,
    # Meta-parameters
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Fused RMSNorm and Linear layer kernel.
    """
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wm + offs_n[None, :] * stride_wn)  # Note: No transpose needed here

    # --- Load data ---
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
    if dtype == tl.float16:
        x = x.to(tl.float32)
        w = w.to(tl.float32)

    # --- RMSNorm ---
    x_sq = x * x
    rms = tl.sum(x_sq, axis=1) / N
    rms = tl.sqrt(rms + eps)
    x_norm = x / rms[:, None]

    # --- Linear Layer ---
    out = tl.dot(x_norm, tl.trans(w)) # Transpose weight for matmul

    # --- Bias (optional) ---
    if HAS_BIAS:
      b_ptrs = b_ptr + offs_k * stride_bm
      b = tl.load(b_ptrs, mask=offs_k < K, other=0.0)
      if dtype == tl.float16:
          b = b.to(tl.float32)
      out += b


    # -- Apply Activation (optional)
    if ACTIVATION == "relu":
        out = tl.where(out > 0, out, 0.0)
    # Add more activations as needed (e.g., gelu, sigmoid)


    # --- Store Output ---
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)

    if dtype == tl.float16:
        out = out.to(tl.float16) # Convert output back to float16

    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))




# PyTorch Reference (RMSNorm + Linear)
class RMSNormLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5, activation=""):
        super().__init__()
        self.rms_norm = RMSNorm(in_features, eps=eps) # Use the RMSNorm class from before
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, x):
        x = self.rms_norm(x)
        x = self.linear(x)
        if self.activation == "relu":
            x = torch.relu(x)
        # Add other activations here as needed
        return x


def run_rmsnorm_linear_test(M, N, K, eps, has_bias, activation, dtype, device='cuda'):
    torch.manual_seed(0)
    x = torch.randn((M, N), device=device, dtype=dtype)
    model_ref = RMSNormLinear(N, K, bias=has_bias, eps=eps, activation=activation).to(device=device, dtype=dtype)
    w = model_ref.linear.weight.data
    b = model_ref.linear.bias.data if has_bias else None


    # Triton Kernel
    out_triton = torch.empty((M, K), device=device, dtype=dtype)
    grid = (triton.cdiv(M, 256), triton.cdiv(K, 32))  # Example grid, adjust as needed
    rmsnorm_linear_kernel[grid](
        x,
        w,
        b if has_bias else x, #Dummy
        out_triton,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        b.stride(0) if has_bias else 0,
        out_triton.stride(0),
        out_triton.stride(1),
        eps=eps,
        BLOCK_SIZE_M=256,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        HAS_BIAS=has_bias,
        ACTIVATION=activation,
        dtype=dtype
    )

    # PyTorch Reference
    out_pytorch = model_ref(x)

    # Comparison
    tolerance = 1e-2 if dtype == torch.float16 else 1e-4

    are_close = torch.allclose(out_triton, out_pytorch, atol=tolerance, rtol=tolerance)
    print(f"M={M}, N={N}, K={K}, dtype={dtype}, Bias={has_bias}, Activation: {activation}, Close: {are_close}")
    if not are_close:
        print("Triton Output:", out_triton)
        print("PyTorch Output:", out_pytorch)
        raise AssertionError("The outputs are not close!")

# Test Cases
test_cases = [
   (64, 128, 256, 1e-5, True, "", torch.float32),
   (32, 512, 1024, 1e-6, False, "", torch.float32),
   (16, 256, 512, 1e-5, True, "relu", torch.float16),
   (8, 1024, 2048, 1e-4, False, "relu", torch.float16),
   (4, 4096, 8192, 1e-5, True, "", torch.float32),  # Large sizes
   (1, 768, 3072, 1e-5, True, "relu", torch.float16), # Single row
    (55, 123, 234, 1e-05, False, "relu", torch.float16), # odd size
    (12, 16, 32, 1e-6, True, "", torch.float32) # small sizes
]

# Prepend RMSNorm definition from previous example
# (so it's available in this scope)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5, has_weight=False):
        super().__init__()
        self.eps = eps
        self.has_weight = has_weight
        if has_weight:
          self.weight = torch.nn.Parameter(torch.ones(dim))
        else:
          self.weight = None #Or just dont make one

    def forward(self, x):
        dtype = x.dtype
        if dtype == torch.float16:
          x = x.to(torch.float32) # Do calculation in FP32
        input_dtype = x.dtype  # Keep track for later
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        if self.weight is not None:
          x = x * self.weight
        return x.to(dtype)  # Return in original dtype.



for M, N, K, eps, has_bias, activation_type, dtype in test_cases:
    run_rmsnorm_linear_test(M, N, K, eps, has_bias, activation_type, dtype)

print("All tests passed!")
