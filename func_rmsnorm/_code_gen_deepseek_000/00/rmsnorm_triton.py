# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    eps, dim_size,
    stride_x_batch, stride_x_dim,
    stride_weight_dim,
    stride_output_batch, stride_output_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    Performs RMS normalization on input tensor rows using Triton.

    Args:
        x_ptr (triton.PointerType): Pointer to input tensor. Shape (batch_size, dim_size)
        weight_ptr (triton.PointerType): Pointer to weight tensor. Shape (dim_size,)
        output_ptr (triton.PointerType): Pointer to output tensor. Shape (batch_size, dim_size)
        eps (float): Small value to avoid division by zero.
        dim_size (int): Dimension size of the input tensor.
        stride_x_batch (int): Stride between batches in the input tensor.
        stride_x_dim (int): Stride between elements in the same batch of the input tensor.
        stride_weight_dim (int): Stride between elements in the weight tensor.
        stride_output_batch (int): Stride between batches in the output tensor.
        stride_output_dim (int): Stride between elements in the same batch of the output tensor.
        BLOCK_SIZE (tl.constexpr): Block size for processing elements.

    Returns:
        None: Output is stored in output_ptr.

    Explanation:
        1. Calculate the program ID to determine the current batch row.
        2. Compute the sum of squares of elements in the row using block-wise reduction.
        3. Compute RMS value using sum and epsilon.
        4. Normalize each element and multiply by weight, then store result.
    """
    pid = tl.program_id(0)
    row_start = pid * stride_x_batch

    # Compute sum of squares
    sum_squares = tl.zeros((1,), dtype=tl.float32)
    for offset in range(0, dim_size, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < dim_size
        x = tl.load(x_ptr + row_start + col_idx * stride_x_dim, mask=mask, other=0.0)
        sum_squares += tl.sum(x * x, axis=0)

    # Calculate RMS
    rms_val = tl.sqrt(sum_squares / dim_size + eps)
    inv_rms = 1.0 / rms_val

    # Apply normalization and weights
    for offset in range(0, dim_size, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < dim_size
        x = tl.load(x_ptr + row_start + col_idx * stride_x_dim, mask=mask, other=0.0)
        w = tl.load(weight_ptr + col_idx * stride_weight_dim, mask=mask, other=0.0)
        normalized = x * inv_rms
        scaled = normalized * w
        tl.store(output_ptr + row_start + col_idx * stride_output_dim, scaled, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS normalization using a Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).
        weight (torch.Tensor): Weight tensor of shape (dim_size,).
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Normalized tensor of shape (batch_size, dim_size).

    Explanation:
        1. Validate input tensor dimensions.
        2. Allocate output tensor with same shape as input.
        3. Launch Triton kernel with grid size equal to batch size.
        4. Return computed output tensor.
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    batch_size, dim_size = x.shape
    output = torch.empty_like(x)
    grid = (batch_size,)
    rmsnorm_kernel[grid](
        x, weight, output,
        eps, dim_size,
        x.stride(0), x.stride(1),
        weight.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=1024
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Applies RMS normalization using PyTorch operations.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size).
        weight (torch.Tensor): Weight tensor of shape (dim_size,).
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Normalized tensor of shape (batch_size, dim_size).

    Explanation:
        1. Compute mean of squared elements along last dimension.
        2. Add epsilon and compute reciprocal square root.
        3. Multiply input by RMS values and weight tensor.
    """
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return x * rms * weight

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps

    verify_correctness(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
