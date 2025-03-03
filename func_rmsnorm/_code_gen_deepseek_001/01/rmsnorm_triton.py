# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    D: int,
    stride_x_batch: int,
    stride_x_dim: int,
    stride_weight: int,
    eps: float,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TRITON kernel for RMS normalization.

    Args:
        x_ptr (tensor): Input tensor pointer [batch_size, dim_size]
        output_ptr (tensor): Output tensor pointer [batch_size, dim_size]
        weight_ptr (tensor): Weight tensor pointer [dim_size]
        D (int): Dimension size of input features
        stride_x_batch (int): Stride between input batches
        stride_x_dim (int): Stride between input features
        stride_weight (int): Stride between weight elements
        eps (float): Numerical stability constant
        BLOCK_SIZE (tl.constexpr): Triton block size for computation
    """
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_batch
    output_row_ptr = output_ptr + row_idx * stride_x_batch

    # 1. Compute sum of squares for the row
    sum_squares = 0.0
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_row_ptr + cols * stride_x_dim, mask=mask, other=0.0)
        x_squared = x * x
        sum_squares += tl.sum(x_squared, axis=0)

    # 2. Calculate RMS value with epsilon
    rms = tl.sqrt((sum_squares / D) + eps)
    inv_rms = 1.0 / rms

    # 3. Normalize and scale with weights
    for offset in range(0, D, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_row_ptr + cols * stride_x_dim, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols * stride_weight, mask=mask, other=0.0)
        normalized = x * inv_rms * weight
        tl.store(output_row_ptr + cols * stride_x_dim, normalized, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor [dim_size]
        eps (float): Numerical stability constant

    Returns:
        torch.Tensor: Normalized output tensor [batch_size, dim_size]

    Explanation:
        1. Validate input shapes
        2. Allocate output tensor
        3. Configure kernel launch parameters
        4. Launch TRITON kernel
    """
    B, D = x.shape
    output = torch.empty_like(x)
    grid = (B,)
    BLOCK_SIZE = triton.next_power_of_2(D)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)  # Cap block size for register limits
    
    _rmsnorm_kernel[grid](
        x, output, weight,
        D,
        x.stride(0), x.stride(1),
        weight.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor [dim_size]
        eps (float): Numerical stability constant

    Returns:
        torch.Tensor: Normalized output tensor [batch_size, dim_size]

    Explanation:
        1. Compute mean of squared elements
        2. Add epsilon and calculate reciprocal root mean square
        3. Apply normalization and weight scaling
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x * (1.0 / rms) * weight

def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    weight = torch.randn(dim_size, device=device, dtype=dtype)
    eps = 1e-5
    return x, weight, eps

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    verify_correctness(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
