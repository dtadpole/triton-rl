# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr,  # Pointer to input tensor [M, N]
    w_ptr,  # Pointer to weight tensor [N]
    y_ptr,  # Pointer to output tensor [M, N]
    stride_xm,  # Stride between rows of input
    stride_yn,  # Stride between columns of output
    N,  # Size of feature dimension
    eps,  # Numerical stability term
    BLOCK_SIZE: tl.constexpr,  # Triton block size (power of two)
):
    """
    Triton kernel for RMS Normalization across feature dimension.

    Args:
        x_ptr (tensor): Input tensor of shape [M, N]
        w_ptr (tensor): Weight tensor of shape [N]
        y_ptr (tensor): Output tensor of shape [M, N]
        stride_xm (int): Stride between input rows
        stride_yn (int): Stride between output columns
        N (int): Feature dimension size
        eps (float): Small value for numerical stability
        BLOCK_SIZE (tl.constexpr): Triton configuration parameter

    Explanation:
        1. Calculate row index for current block
        2. Compute memory offsets for input/output access
        3. Iteratively compute sum of squares across feature dimension
        4. Use parallel reduction to calculate total squared sum
        5. Compute RMS value with epsilon stabilization
        6. Normalize and scale inputs using weights
    """
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * stride_xm + tl.arange(0, BLOCK_SIZE)
    w_ptrs = w_ptr + tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum and load pointers
    sum_squares = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(x_ptrs + cols * stride_yn, mask=mask, other=0.0)
        x_sq = x * x
        sum_squares += tl.sum(x_sq, axis=0)

    # Calculate RMS value
    mean_squares = sum_squares / N
    rms = 1.0 / tl.sqrt(mean_squares + eps)

    # Apply normalization and weights
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        
        x = tl.load(x_ptrs + cols * stride_yn, mask=mask)
        w = tl.load(w_ptrs + cols, mask=mask)
        y = x * rms * w
        
        tl.store(y_ptr + row * stride_yn + cols * stride_yn, y, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton implementation of RMS Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of shape [batch_size, dim_size]

    Explanation:
        1. Validate input dimensions
        2. Allocate output tensor
        3. Determine optimal block size (nearest power of two <= dim_size)
        4. Launch Triton kernel with calculated grid configuration
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    M, N = x.shape
    y = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    
    rmsnorm_kernel[(M,)](
        x, weight, y,
        x.stride(0), y.stride(1),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size]
        weight (torch.Tensor): Weight tensor of shape [dim_size]
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of shape [batch_size, dim_size]

    Explanation:
        1. Compute mean of squared values along last dimension
        2. Add epsilon for numerical stability
        3. Compute reciprocal square root of adjusted mean
        4. Multiply normalized values with weights
    """
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        weight = torch.randn(dim_size, device=device, dtype=dtype)
        eps = 1e-5
        return x, weight, eps
    
    verify_correctness(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(rmsnorm_triton, rmsnorm_torch, input_generator, [batch_sizes, dim_sizes])
