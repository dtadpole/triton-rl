# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['dim_size']
)
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    dim_size, eps,
    stride_x_batch, stride_x_dim,
    stride_weight_dim,
    stride_output_batch, stride_output_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute RMS normalization for input tensor using Triton.

    Args:
        x_ptr (tensor): Pointer to input tensor of shape (batch_size, dim_size)
        weight_ptr (tensor): Pointer to weight tensor of shape (dim_size,)
        output_ptr (tensor): Pointer to output tensor of shape (batch_size, dim_size)
        dim_size (int): Size of the feature dimension
        eps (float): Small value for numerical stability
        stride_x_batch (int): Stride for batch dimension in input tensor
        stride_x_dim (int): Stride for feature dimension in input tensor
        stride_weight_dim (int): Stride for weight tensor
        stride_output_batch (int): Stride for batch dimension in output tensor
        stride_output_dim (int): Stride for feature dimension in output tensor
        BLOCK_SIZE (tl.constexpr): Block size for Triton kernel

    Returns:
        None: Output is written directly to output_ptr

    Explanation:
        1. Calculate the position of the current batch
        2. Load a block of data from the input tensor
        3. Compute squared values and perform block-wise reduction
        4. Calculate RMS value using the reduced sum
        5. Normalize input and scale with weight
        6. Store normalized values to output tensor
    """
    # Position in batch dimension
    batch_idx = tl.program_id(0)
    
    x_batch_ptr = x_ptr + batch_idx * stride_x_batch
    output_batch_ptr = output_ptr + batch_idx * stride_output_batch
    
    # Compute mean squared value using reduction
    sum_squares = tl.zeros((1,), dtype=tl.float32)
    
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        
        x_ptr = x_batch_ptr + cols * stride_x_dim
        x = tl.load(x_ptr, mask=mask, other=0.0)
        
        sum_squares += tl.sum(x * x, axis=0)
    
    rms = tl.sqrt(sum_squares / dim_size + eps)
    inv_rms = 1.0 / rms

    # Normalize and scale with weights
    for offset in range(0, dim_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        
        x_ptr = x_batch_ptr + cols * stride_x_dim
        weight_ptr = weight_ptr + cols * stride_weight_dim
        
        x = tl.load(x_ptr, mask=mask)
        weight = tl.load(weight_ptr, mask=mask)
        
        normalized = x * inv_rms
        scaled = normalized * weight
        
        output_ptr = output_batch_ptr + cols * stride_output_dim
        tl.store(output_ptr, scaled, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized tensor of same shape as input

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Launch Triton kernel with appropriate grid configuration
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    batch_size, dim_size = x.shape
    
    output = torch.empty_like(x)
    
    grid = (batch_size,)
    
    rmsnorm_kernel[grid](
        x, weight, output,
        dim_size, eps,
        x.stride(0), x.stride(1),
        weight.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=min(triton.next_power_of_2(dim_size), 1024)
    )
    
    return output


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized tensor of same shape as input

    Explanation:
        1. Compute mean squared value along feature dimension
        2. Calculate inverse RMS value
        3. Normalize input and scale with weight
    """
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    inv_rms = 1.0 / rms
    return x * inv_rms * weight


def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    weight = torch.randn(dim_size, device=device, dtype=dtype)
    eps = 1e-5
    return x, weight, eps


if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
