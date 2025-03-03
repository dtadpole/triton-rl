# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    weight_ptr,  # Pointer to weight tensor
    stride_x,  # Stride for input tensor dimensions
    stride_weight,  # Stride for weight tensor
    N,  # Size of normalized dimension
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size for kernel execution
):
    """
    Triton kernel for RMS Normalization computation.

    Args:
        x_ptr (tl.tensor): Pointer to input tensor of shape (B, N)
        output_ptr (tl.tensor): Pointer to output tensor of shape (B, N)
        weight_ptr (tl.tensor): Pointer to weight tensor of shape (N,)
        stride_x (int): Stride between rows in input tensor
        stride_weight (int): Stride for weight tensor (unused in this implementation)
        N (int): Size of the normalized dimension
        eps (float): Small value to prevent division by zero
        BLOCK_SIZE (tl.constexpr): Block size configuration for kernel optimization

    Returns:
        None: Output is written directly to output_ptr

    Explanation:
        1. Calculate row index using program ID
        2. Compute memory offsets for current row
        3. Load input data into shared memory with masking
        4. Calculate sum of squares for normalization
        5. Compute RMS value with epsilon stabilization
        6. Normalize and scale values with weight matrix
        7. Store results in output tensor
    """
    row = tl.program_id(0)
    x_offsets = row * stride_x + tl.arange(0, BLOCK_SIZE)
    weight_offsets = tl.arange(0, BLOCK_SIZE)

    mask = weight_offsets < N
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)

    x_float = x.to(tl.float32)
    var = tl.sum(x_float * x_float, axis=0) / N
    rms = tl.sqrt(var + eps)
    inv_rms = 1.0 / rms

    normalized = (x_float * inv_rms).to(x.dtype)
    weighted = normalized * weight

    tl.store(output_ptr + x_offsets, weighted, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    TRITON implementation of RMS Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (B, D)
        weight (torch.Tensor): Weight tensor of shape (D,)
        eps (float): Stabilization constant (default: 1e-5)

    Returns:
        torch.Tensor: Normalized output tensor of shape (B, D)

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Determine optimal block size
        4. Launch TRITON kernel with calculated grid configuration
    """
    B, D = x.shape
    output = torch.empty_like(x)
    
    # Configure kernel execution parameters
    BLOCK_SIZE = triton.next_power_of_2(D)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
        
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8

    grid = (B,)
    rmsnorm_kernel[grid](
        x, output, weight,
        x.stride(0), weight.stride(0),
        N=D,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return output


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (B, D)
        weight (torch.Tensor): Weight tensor of shape (D,)
        eps (float): Stabilization constant (default: 1e-5)

    Returns:
        torch.Tensor: Normalized output tensor of shape (B, D)

    Explanation:
        1. Compute root mean square along last dimension
        2. Normalize input using RMS value
        3. Apply learned weight scaling
    """
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    x_norm = x / rms
    return x_norm * weight


# Testing and benchmarking configuration
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
