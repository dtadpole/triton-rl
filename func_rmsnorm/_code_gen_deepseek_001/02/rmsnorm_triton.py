# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd(
    x_ptr,
    output_ptr,
    weight_ptr,
    eps,
    n_cols,
    stride_x_batch,
    stride_x_col,
    stride_weight,
    stride_output_batch,
    stride_output_col,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes RMS normalization for a batch of inputs using Triton.

    Args:
        x_ptr (Pointer): Input tensor of shape (batch_size, dim_size)
        output_ptr (Pointer): Output tensor of shape (batch_size, dim_size)
        weight_ptr (Pointer): Weight tensor of shape (dim_size,)
        eps (float): Small value to prevent division by zero
        n_cols (int): Number of columns (feature dimension size)
        stride_x_batch (int): Stride between batch elements in input
        stride_x_col (int): Stride between columns in input
        stride_weight (int): Stride for weight tensor elements
        stride_output_batch (int): Stride between batch elements in output
        stride_output_col (int): Stride between columns in output
        BLOCK_SIZE (tl.constexpr): Triton block size for kernel execution

    Returns:
        None: Output is stored directly in output_ptr

    Explanation:
        1. Calculate program ID for current batch element
        2. Initialize pointers for input/output batches
        3. Compute sum of squares in chunks to handle large dimensions
        4. Calculate RMS value from summed squares
        5. Apply normalization and scaling in second pass over elements
        6. Store normalized values using fused multiply operations
    """
    pid = tl.program_id(0)
    x_batch_ptr = x_ptr + pid * stride_x_batch
    output_batch_ptr = output_ptr + pid * stride_output_batch

    # First pass: compute sum of squares
    sum_x_sq = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        x = tl.load(x_batch_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        sum_x_sq += tl.sum(x * x, axis=0)

    variance = sum_x_sq / n_cols
    rsqrt_var = 1.0 / tl.sqrt(variance + eps)

    # Second pass: normalize and scale
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        
        x = tl.load(x_batch_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols * stride_weight, mask=mask, other=0.0).to(tl.float32)
        
        output = x * rsqrt_var * weight
        tl.store(output_batch_ptr + cols * stride_output_col, output.to(tl.float16), mask=mask)

def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Triton frontend for RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of shape (batch_size, dim_size)

    Explanation:
        1. Validate input dimensions
        2. Determine optimal block size (power of two up to 1024)
        3. Allocate output tensor
        4. Configure and launch Triton kernel
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    assert weight.dim() == 1, "Weight must be 1D tensor"
    assert x.size(1) == weight.size(0), "Feature dimension mismatch"

    batch, dim = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(dim), 1024)
    
    output = torch.empty_like(x)
    grid = (batch,)
    
    _rms_norm_fwd[grid](
        x, output, weight, eps, dim,
        x.stride(0), x.stride(1),
        weight.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    PyTorch reference implementation of RMS normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
        weight (torch.Tensor): Weight tensor of shape (dim_size,)
        eps (float): Small value for numerical stability

    Returns:
        torch.Tensor: Normalized output tensor of shape (batch_size, dim_size)

    Explanation:
        1. Compute mean squared value across feature dimension
        2. Calculate normalization scale factor
        3. Apply normalization and weight scaling
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

    verify_correctness(rms_norm_triton, rms_norm_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(rms_norm_triton, rms_norm_torch, input_generator, [batch_sizes, dim_sizes])
