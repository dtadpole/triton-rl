# filename: rmsnorm_triton.py

import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for RMS Normalization.

    Args:
        x_ptr (tl.tensor): Pointer to input tensor of shape (B, N)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (N,)
        output_ptr (tl.tensor): Pointer to output tensor of shape (B, N)
        N (int): Feature dimension size
        eps (float): Small value for numerical stability
        BLOCK_SIZE (tl.constexpr): Triton block size

    Explanation:
        1. Each program processes one row of the input matrix
        2. Load data from global memory into SRAM
        3. Compute sum of squares for normalization
        4. Apply normalization and weight scaling
        5. Write results back to global memory
    """
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    
    # Load input row with boundary check
    x_ptrs = x_ptr + row_idx * N + cols
    mask = cols < N
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute variance
    x_float = tl.math.astype(x, tl.float32)
    sum_sq = tl.sum(x_float * x_float, axis=0) / N
    rms = tl.math.rsqrt(sum_sq + eps)
    
    # Normalize and scale
    normalized = x_float * rms
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    output = normalized * tl.math.astype(weight, tl.float32)
    
    # Write back results
    tl.store(x_ptrs, tl.math.astype(output, x.dtype), mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of same shape as input

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Configure kernel launch parameters
        4. Launch RMSNorm kernel
    """
    B, N = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
        
    grid = (B,)
    rmsnorm_kernel[grid](
        x, weight, output,
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of same shape as input

    Explanation:
        1. Compute root mean square along feature dimension
        2. Normalize input using computed RMS
        3. Apply learned weight scaling
    """
    rms = torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    return (x * rms).to(x.dtype) * weight

if __name__ == "__main__":
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
