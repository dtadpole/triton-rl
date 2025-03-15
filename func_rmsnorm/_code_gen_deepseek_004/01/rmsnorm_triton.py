# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor, shape (B, D)
    w_ptr,  # Pointer to weight tensor, shape (D,)
    y_ptr,  # Pointer to output tensor, shape (B, D)
    eps,    # Epsilon for numerical stability
    B,      # Batch size
    D,      # Feature dimension size
    stride_x_batch,  # Stride between batch elements in x
    stride_x_dim,    # Stride between features in x
    stride_w_dim,    # Stride between features in weight (typically 1)
    stride_y_batch,  # Stride between batch elements in y
    stride_y_dim,    # Stride between features in y
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute RMS normalization for a batch of vectors using Triton.

    Args:
        x_ptr (triton.Pointer): Input tensor pointer [B, D]
        w_ptr (triton.Pointer): Weight tensor pointer [D]
        y_ptr (triton.Pointer): Output tensor pointer [B, D]
        eps (float): Small value to avoid division by zero
        B (int): Batch size dimension
        D (int): Feature dimension size
        stride_x_batch (int): Stride between x batch elements
        stride_x_dim (int): Stride between x features
        stride_w_dim (int): Stride between weight features
        stride_y_batch (int): Stride between y batch elements
        stride_y_dim (int): Stride between y features
        BLOCK_SIZE (tl.constexpr): Block size for Triton operations
    """
    # Get current program's batch index
    batch_idx = tl.program_id(0)
    
    # Calculate pointers to current batch in input and output
    x_batch_ptr = x_ptr + batch_idx * stride_x_batch
    y_batch_ptr = y_ptr + batch_idx * stride_y_batch
    
    # Initialize sum of squares
    sum_sq = 0.0
    
    # Compute sum of squares in blocks
    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        
        # Load input values and compute squares
        x = tl.load(x_batch_ptr + col * stride_x_dim, mask=mask, other=0.0)
        x_sq = x * x
        sum_sq += tl.sum(x_sq, axis=0)
    
    # Compute RMS value
    rms = tl.sqrt(sum_sq / D + eps)
    
    # Normalize and scale with weight
    for offset in range(0, D, BLOCK_SIZE):
        col = offset + tl.arange(0, BLOCK_SIZE)
        mask = col < D
        
        # Load input and weight
        x = tl.load(x_batch_ptr + col * stride_x_dim, mask=mask, other=0.0)
        w = tl.load(w_ptr + col * stride_w_dim, mask=mask, other=0.0)
        
        # Compute normalized output
        y = (x / rms) * w
        
        # Store result
        tl.store(y_batch_ptr + col * stride_y_dim, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Output tensor of same shape as input

    Explanation:
        1. Verify input dimensions and contiguity
        2. Allocate output tensor
        3. Determine optimal block size (up to 1024)
        4. Launch Triton kernel with appropriate grid configuration
        5. Return normalized and scaled output
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    B, D = x.shape
    assert weight.shape == (D,), "Weight shape mismatch"
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Configure kernel launch parameters
    BLOCK_SIZE = triton.next_power_of_2(D)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    grid = (B,)
    
    # Launch kernel
    rmsnorm_kernel[grid](
        x, weight, y, eps,
        B, D,
        x.stride(0), x.stride(1),
        weight.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Output tensor of same shape as input

    Explanation:
        1. Compute mean of squared values along feature dimension
        2. Add epsilon and compute reciprocal square root
        3. Normalize input values
        4. Scale with weight vector
    """
    rms = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x * (weight / (rms + eps))


if __name__ == "__main__":
    # Configuration parameters from problem statement
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
