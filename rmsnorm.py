# filename: rmsnorm.py
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride_xn,  # how many elements to advance to get to the next row in X
    stride_yn,  # how many elements to advance to get to the next row in Y
    stride_wn,  # how many elements to advance to get to the next row in W
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the elements of X, Y, and W that it should compute.
    row = tl.program_id(0)
    X += row * stride_xn
    Y += row * stride_yn
    W += row * stride_wn

    # Compute the sum of squares of the elements of X in parallel.
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)  # Load weights
    x_sq = x * x
    # tl.sum with axis=0 does a reduction along the columns
    x_sq_sum = tl.sum(x_sq, axis=0)
    # Compute rms
    inv_rms = 1.0 / tl.sqrt(x_sq_sum / N + eps)

    # Normalize and scale the elements of X
    y = x * inv_rms * w
    tl.store(Y + cols, y, mask=mask)


class RMSNorm(torch.nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5, dtype=torch.float32):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=dtype))

    def forward(self, x):
        if x.dtype != self.weight.dtype: # type: ignore
            x = x.to(self.weight.dtype)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        y = torch.empty_like(x_arg)
        # Assume BLOCK_SIZE is a power of 2, can be tuned
        BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
        grid = (M, )
        _rmsnorm_fwd_kernel[grid](
            x_arg,
            y,
            self.weight,
            x_arg.stride(0),
            y.stride(0),
            self.weight.stride(0),
            N,
            self.eps,
            BLOCK_SIZE,
        )
        return y.reshape(x.shape)



class TorchRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype = torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if x.dtype != self.weight.dtype: # type: ignore
            x = x.to(self.weight.dtype)
        output = self._norm(x)
        return output * self.weight

