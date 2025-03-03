# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    x_ptr, output_ptr, weight_ptr,
    N: tl.constexpr, eps: tl.constexpr,
    x_row_stride: tl.constexpr, output_row_stride: tl.constexpr,
    weight_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes RMS normalization for input tensor using Triton.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer. Shape (batch, dim)
        output_ptr (triton.Pointer): Output tensor pointer. Shape (batch, dim)
        weight_ptr (triton.Pointer): Weight vector pointer. Shape (dim,)
        N (tl.constexpr): Dimension size of input
        eps (tl.constexpr): Numerical stability epsilon
        x_row_stride (tl.constexpr): Stride between input rows (elements)
        output_row_stride (tl.constexpr): Stride between output rows (elements)
        weight_row_stride (tl.constexpr): Unused (maintained for API consistency)
        BLOCK_SIZE (tl.constexpr): Triton processing block size

    Returns:
        None: Output written directly to output_ptr

    Explanation:
        1. Calculate row index from program ID
        2. Initialize sum of squares accumulator
        3. Load data in blocks, accumulate squared values
        4. Compute RMS value from sum and dimension
        5. Normalize inputs and apply weights in blocks
    """
    row_idx = tl.program_id(0)
    x_row_start = x_ptr + row_idx * x_row_stride
    output_row_start = output_ptr + row_idx * output_row_stride

    # Sum squares reduction
    sum_sq = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_start + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    rms = tl.sqrt(sum_sq / N + eps)

    # Normalize and weight application
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_start + cols, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        normalized = x / rms * weight
        tl.store(output_row_start + cols, normalized, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS normalization using Triton kernel.

    Args:
        x (torch.Tensor): Input tensor. Shape (batch, dim)
        weight (torch.Tensor): Weight vector. Shape (dim,)
        eps (float): Numerical stability term. Default 1e-5

    Returns:
        torch.Tensor: Normalized output tensor. Shape (batch, dim)

    Explanation:
        1. Validate input dimensions and contiguity
        2. Preallocate output tensor
        3. Configure kernel grid and block parameters
        4. Launch kernel with appropriate memory pointers
    """
    assert x.shape[-1] == weight.shape[0], "Input dim mismatch with weight"
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert weight.is_contiguous(), "Weight tensor must be contiguous"
    
    output = torch.empty_like(x)
    N = x.shape[-1]
    grid = (x.shape[0],)
    BLOCK_SIZE = 1024  # Optimal for most modern GPUs

    _rmsnorm_kernel[grid](
        x, output, weight,
        N, eps,
        x.stride(0), output.stride(0),
        weight.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Applies RMS normalization using PyTorch operations.

    Args:
        x (torch.Tensor): Input tensor. Shape (batch, dim)
        weight (torch.Tensor): Weight vector. Shape (dim,)
        eps (float): Numerical stability term. Default 1e-5

    Returns:
        torch.Tensor: Normalized output tensor. Shape (batch, dim)

    Explanation:
        1. Compute mean of squared values along feature dimension
        2. Add epsilon and compute reciprocal square root
        3. Element-wise multiplication with weight vector
    """
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    x_hat = x / rms
    return x_hat * weight

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
