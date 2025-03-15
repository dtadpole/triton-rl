# filename: rmsnorm_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_rows, n_cols,
    stride_xrow, stride_xcol,
    stride_w, 
    stride_yrow, stride_ycol,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMSNorm.

    Args:
        x_ptr: Pointer to the input tensor. [batch_size, dim_size]
        weight_ptr: Pointer to the weight tensor. [dim_size]
        output_ptr: Pointer to the output tensor. [batch_size, dim_size]
        n_rows: Number of rows in the input.
        n_cols: Number of columns in the input.
        stride_xrow: Stride between rows in the input.
        stride_xcol: Stride between columns in the input.
        stride_w: Stride for the weight tensor.
        stride_yrow: Stride between rows in the output.
        stride_ycol: Stride between columns in the output.
        eps: Small constant for numerical stability.
        BLOCK_SIZE: Size of the block for processing.

    Explanation:
        1. Each thread block processes one row of the input.
        2. Load the input row data with appropriate masking.
        3. Calculate the squared values of each element in the row.
        4. Compute the sum of squared values across all elements in the row.
        5. Calculate the RMS value (root mean square).
        6. Load the weight tensor.
        7. Normalize the row by dividing by the RMS value and scale by weight.
        8. Store the normalized and scaled row to the output tensor.
    """
    # Get the program ID (row index)
    row_idx = tl.program_id(0)
    
    # Compute pointer to current row
    x_row_ptr = x_ptr + row_idx * stride_xrow
    output_row_ptr = output_ptr + row_idx * stride_yrow
    
    # Load the row data
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    x_cols = tl.load(x_row_ptr + col_offsets * stride_xcol, mask=mask, other=0.0)
    
    # Calculate the RMS for normalization
    x_cols_squared = x_cols * x_cols
    row_sum_squared = tl.sum(x_cols_squared)
    row_mean_squared = row_sum_squared / n_cols
    row_rms = tl.sqrt(row_mean_squared + eps)
    
    # Load the weight and normalize
    weight = tl.load(weight_ptr + col_offsets * stride_w, mask=mask, other=0.0)
    output = (x_cols / row_rms) * weight
    
    # Store the result
    tl.store(output_row_ptr + col_offsets * stride_ycol, output, mask=mask)

def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton implementation of RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size].
        weight (torch.Tensor): Scale parameter of shape [dim_size].
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Normalized tensor of shape [batch_size, dim_size].

    Explanation:
        1. Determine the block size for Triton kernel based on dimension size.
        2. Launch the kernel to process each row in parallel.
        3. Each row is normalized by its own RMS value.
        4. Return the normalized and scaled tensor.
    """
    batch_size, dim_size = x.shape
    output = torch.empty_like(x)
    
    # Determine block size (power of 2 close to dim_size)
    BLOCK_SIZE = triton.next_power_of_2(dim_size)
    
    # Launch the kernel
    grid = (batch_size,)
    _rmsnorm_kernel[grid](
        x, weight, output,
        batch_size, dim_size,
        x.stride(0), x.stride(1),
        weight.stride(0),
        output.stride(0), output.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    PyTorch implementation of RMSNorm.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim_size].
        weight (torch.Tensor): Scale parameter of shape [dim_size].
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Normalized tensor of shape [batch_size, dim_size].

    Explanation:
        1. Calculate the squared values of each element in the input.
        2. Compute the mean of the squared values along the feature dimension.
        3. Take the square root to get the RMS value.
        4. Normalize each element by dividing by the RMS value.
        5. Scale the normalized values by the learnable parameters.
    """
    # Calculate the root mean square value
    variance = torch.mean(x ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(variance + eps)
    
    # Normalize and scale
    x_normalized = x / rms
    return x_normalized * weight

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
