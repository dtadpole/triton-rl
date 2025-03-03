# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    dim_size: tl.int32,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Performs RMS normalization on input tensor using Triton.

    Args:
        x_ptr (triton.PointerType): Input tensor pointer [shape: (batch, dim_size)]
        y_ptr (triton.PointerType): Output tensor pointer [shape: (batch, dim_size)]
        w_ptr (triton.PointerType): Weight tensor pointer [shape: (dim_size,)]
        dim_size (tl.int32): Feature dimension size
        eps (tl.float32): Numerical stability epsilon
        BLOCK_SIZE (tl.constexpr): Block size for element processing

    Returns:
        None: Output stored through y_ptr

    Explanation:
        1. Calculate row index from program ID
        2. Compute sum of squares in blocks to avoid memory overload
        3. Calculate RMS value from squared sum
        4. Apply normalization and weight scaling in blocks
    """
    pid = tl.program_id(0)
    row_start = pid * dim_size
    sum_squares = 0.0

    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(x_ptr + row_start + cols, mask, 0.0)
        sum_squares += tl.sum(x * x, axis=0)

    rms = 1.0 / tl.sqrt((sum_squares / dim_size) + eps)

    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(x_ptr + row_start + cols, mask, 0.0)
        w = tl.load(w_ptr + cols, mask, 0.0)
        tl.store(y_ptr + row_start + cols, x * rms * w, mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton wrapper function for RMS normalization.

    Args:
        x (torch.Tensor): Input tensor [shape: (batch, dim_size)]
        weight (torch.Tensor): Weight tensor [shape: (dim_size,)]
        eps (float): Numerical stability epsilon

    Returns:
        torch.Tensor: Normalized output tensor [shape: (batch, dim_size)]

    Explanation:
        1. Ensure contiguous memory layout
        2. Validate tensor dimensions
        3. Configure kernel launch parameters
        4. Launch RMSNorm kernel
    """
    x = x.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x)
    batch, dim = x.shape
    assert weight.shape[0] == dim, "Weight dimension mismatch"
    
    grid = (batch,)
    rmsnorm_kernel[grid](x, y, weight, dim, eps, BLOCK_SIZE=1024)
    return y

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor [shape: (batch, dim_size)]
        weight (torch.Tensor): Weight tensor [shape: (dim_size,)]
        eps (float): Numerical stability epsilon

    Returns:
        torch.Tensor: Normalized output tensor [shape: (batch, dim_size)]

    Explanation:
        1. Compute mean of squared values
        2. Calculate RMS scaling factor
        3. Apply normalization and weight scaling
    """
    rms = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt().reciprocal()
    return x * rms * weight

# Benchmark configuration
batch_sizes = [256, 512, 1024, 2048]
dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    weight = torch.randn(dim_size, device=device, dtype=dtype)
    eps = 1e-5
    return x, weight, eps

from benchmark import verify_correctness_func, benchmark_performance_func

if __name__ == "__main__":
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
