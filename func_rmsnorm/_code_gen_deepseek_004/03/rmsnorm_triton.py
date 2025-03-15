# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Scale tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized tensor of same shape as input

    Explanation:
        1. Compute squared values of input
        2. Calculate mean of squares along last dimension
        3. Normalize input using root mean square + epsilon
        4. Scale normalized values with weight vector
    """
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight

@triton.jit
def _rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    dim_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMS normalization.

    Args:
        x_ptr (tl.tensor): Input tensor pointer of shape (batch_size, dim_size)
        weight_ptr (tl.tensor): Weight tensor pointer of shape (dim_size,)
        output_ptr (tl.tensor): Output tensor pointer
        dim_size (tl.constexpr): Dimension size of input
        eps (tl.constexpr): Numerical stability term
        BLOCK_SIZE (tl.constexpr): Block size for parallel computation

    Explanation:
        1. Load input and weight data in blocks
        2. Compute sum of squares for normalization
        3. Calculate normalization scale factor
        4. Apply normalization and scaling
    """
    row_idx = tl.program_id(0)
    offsets = row_idx * dim_size + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < dim_size

    # Load input and weights
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # Compute sum of squares
    x_square = x * x
    sum_squares = tl.sum(x_square, axis=0) / dim_size

    # Calculate normalization factor
    rms = tl.math.rsqrt(sum_squares + eps)
    
    # Normalize and scale
    output = x * rms * weight
    tl.store(output_ptr + offsets, output, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton-accelerated RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Scale tensor of shape (dim_size,)
        eps (float): Numerical stability term

    Returns:
        torch.Tensor: Normalized output tensor of same shape as input

    Explanation:
        1. Validate input dimensions
        2. Determine optimal block size
        3. Launch kernel with appropriate grid configuration
    """
    batch_size, dim_size = x.shape
    assert weight.shape == (dim_size,), "Weight shape mismatch"

    # Determine block size (next power of two up to 1024)
    BLOCK_SIZE = triton.next_power_of_2(dim_size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    output = torch.empty_like(x)
    grid = (batch_size,)
    
    _rmsnorm_kernel[grid](
        x, weight, output,
        dim_size=dim_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

if __name__ == "__main__":
    # Benchmark configurations
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps

    from benchmark import verify_correctness_func, benchmark_performance_func

    # Verify numerical correctness
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])

    # Benchmark performance
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
