# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    dim_size,
    eps,
    stride_x_row,
    stride_weight,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes RMS normalization for input tensor using Triton.

    Args:
        x_ptr (tl.tensor): Input tensor of shape (batch_size, dim_size).
        weight_ptr (tl.tensor): Weight tensor of shape (dim_size).
        output_ptr (tl.tensor): Output tensor of shape (batch_size, dim_size).
        dim_size (int): Dimension size of the input and weight tensors.
        eps (float): Small value added to variance for numerical stability.
        stride_x_row (int): Stride between rows in input tensor.
        stride_weight (int): Stride for weight tensor (should be 1).
        stride_output_row (int): Stride between rows in output tensor.
        BLOCK_SIZE (tl.constexpr): Triton block size (must be power of two).

    Returns:
        None: Output is stored in output_ptr.

    Explanation:
        1. Each block processes one row from the input tensor.
        2. First loop computes sum of squares of elements in row.
        3. Calculate RMS using sum, dim_size and epsilon.
        4. Second loop normalizes elements and applies weight.
    """
    row_idx = tl.program_id(0)
    x_row_ptr = x_ptr + row_idx * stride_x_row
    output_row_ptr = output_ptr + row_idx * stride_output_row

    # Compute sum of squares
    sum_sq = 0.0
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    rms = tl.sqrt(sum_sq / dim_size + eps)

    # Apply normalization and weights
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        normalized = (x / rms) * w
        tl.store(output_row_ptr + cols, normalized, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS normalization using Triton kernel.

    Args:
        x (Tensor): Input tensor of shape (batch_size, dim_size).
        weight (Tensor): Weight tensor of shape (dim_size).
        eps (float): Epsilon for numerical stability.

    Returns:
        Tensor: Normalized output tensor of same shape as input.

    Explanation:
        1. Verifies input dimensions match weight shape.
        2. Initializes output tensor.
        3. Configures kernel launch parameters.
        4. Launches Triton kernel with optimized block size.
    """
    output = torch.empty_like(x)
    batch_size, dim_size = x.shape
    
    # Adjust BLOCK_SIZE to largest power-of-two <= dim_size
    max_block_size = 1024  # Triton's maximum recommended block size
    BLOCK_SIZE = min(max_block_size, triton.next_power_of_2(dim_size))
    
    grid = (batch_size,)
    rmsnorm_kernel[grid](
        x, weight, output,
        dim_size,
        eps,
        x.stride(0),
        weight.stride(0),
        output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch implementation of RMS normalization.

    Args:
        x (Tensor): Input tensor of shape (batch_size, dim_size).
        weight (Tensor): Weight tensor of shape (dim_size).
        eps (float): Epsilon for numerical stability.

    Returns:
        Tensor: Normalized output tensor of same shape as input.

    Explanation:
        1. Compute mean of squared values along last dimension.
        2. Add epsilon and compute reciprocal square root.
        3. Multiply normalized values by weight element-wise.
    """
    rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x * (weight / (rms + eps))

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
