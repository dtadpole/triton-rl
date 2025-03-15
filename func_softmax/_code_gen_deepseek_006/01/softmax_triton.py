# filename: softmax_triton.py
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def softmax_kernel(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    input_row_stride: int,
    output_row_stride: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr
):
    """
    Applies softmax to a single row of the input tensor using Triton.

    Args:
        output_ptr (tl.tensor): Output tensor pointer [batch_size, dim_size]
        input_ptr (tl.tensor): Input tensor pointer [batch_size, dim_size]
        input_row_stride (int): Stride between rows in input tensor
        output_row_stride (int): Stride between rows in output tensor
        n_cols (int): Number of columns in the tensor
        BLOCK_SIZE (tl.constexpr): Block size for parallel computation

    Returns:
        None: Output is stored directly through pointers

    Explanation:
        1. Calculate current row index using program ID
        2. Generate column offsets within the block
        3. Load input row data with boundary checking
        4. Compute maximum value for numerical stability
        5. Subtract max and compute exponentials
        6. Calculate sum of exponentials
        7. Normalize and store results
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load row data with padding handling
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    
    # Compute numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store results
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax to a 2D tensor using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied, same shape as input

    Explanation:
        1. Validate input dimensions
        2. Determine optimal block size (next power of two of columns)
        3. Allocate output tensor
        4. Launch kernel with appropriate grid/block configuration
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    n_rows, n_cols = x.shape
    
    # Determine block size as next power of two of columns
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(64, min(BLOCK_SIZE, 4096))  # Clamp for GPU limits
    
    # Allocate output and launch kernel
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax using PyTorch's implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied, same shape as input

    Explanation:
        1. Directly uses PyTorch's optimized softmax implementation
        2. Specifies dimension for normalization (last dimension)
    """
    return F.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
