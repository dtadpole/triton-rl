# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr, output_ptr,
    n_rows, n_cols,
    stride_row_x, stride_col_x,
    stride_row_output, stride_col_output,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute the softmax function along the last dimension of the input tensor.

    Args:
        x_ptr (tl.tensor): Pointer to the input tensor. [batch_size, dim_size]
        output_ptr (tl.tensor): Pointer to the output tensor. [batch_size, dim_size]
        n_rows (int): Number of rows in the input tensor.
        n_cols (int): Number of columns in the input tensor.
        stride_row_x (int): Stride for moving one row in the input tensor.
        stride_col_x (int): Stride for moving one column in the input tensor.
        stride_row_output (int): Stride for moving one row in the output tensor.
        stride_col_output (int): Stride for moving one column in the output tensor.
        BLOCK_SIZE (tl.constexpr): Size of the processing block.

    Returns:
        None: The result is written directly to output_ptr.

    Explanation:
        1. Load a row of the input tensor.
        2. Compute the maximum value of the row for numerical stability.
        3. Compute the exponential of each element minus the maximum.
        4. Compute the sum of the exponentials.
        5. Divide each exponential by the sum to get the softmax values.
        6. Store the softmax values in the output tensor.
    """
    # Get the program id (corresponds to row index)
    row_idx = tl.program_id(0)
    
    # Compute the memory address of the elements to load
    row_start_x = row_idx * stride_row_x
    row_start_output = row_idx * stride_row_output
    
    # Compute column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load the row
    cols_x = row_start_x + col_offsets * stride_col_x
    row = tl.load(x_ptr + cols_x, mask=mask, other=-float('inf'))
    
    # Compute the maximum value for numerical stability
    row_max = tl.max(row)
    
    # Apply the exponential function and subtract the maximum
    numerator = tl.exp(row - row_max)
    
    # Compute the sum for the denominator
    denominator = tl.sum(numerator)
    
    # Compute the softmax
    softmax_output = numerator / denominator
    
    # Store the result
    cols_output = row_start_output + col_offsets * stride_col_output
    tl.store(output_ptr + cols_output, softmax_output, mask=mask)

def softmax_triton(x):
    """
    Compute softmax along the last dimension using a Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size). [batch_size, dim_size]

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim_size) containing softmax values. [batch_size, dim_size]

    Explanation:
        1. Get the shape and strides of the input tensor.
        2. Allocate the output tensor with the same shape and dtype.
        3. Determine the block size for the Triton kernel.
        4. Launch the Triton kernel to compute softmax for each row.
        5. Return the output tensor.
    """
    # Get the input shape
    batch_size, dim_size = x.shape
    
    # Allocate the output tensor
    output = torch.empty_like(x)
    
    # Get the strides
    stride_row_x = x.stride(0)
    stride_col_x = x.stride(1)
    stride_row_output = output.stride(0)
    stride_col_output = output.stride(1)
    
    # Round up to the nearest multiple of BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(dim_size)
    
    # Launch the kernel
    softmax_kernel[(batch_size,)](
        x, output,
        batch_size, dim_size,
        stride_row_x, stride_col_x,
        stride_row_output, stride_col_output,
        BLOCK_SIZE
    )
    
    return output

def softmax_torch(x):
    """
    Compute softmax along the last dimension using PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size). [batch_size, dim_size]

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, dim_size) containing softmax values. [batch_size, dim_size]

    Explanation:
        1. Use PyTorch's built-in softmax function to compute softmax along the last dimension.
    """
    return torch.softmax(x, dim=-1)

# Name both generated functions for benchmark
generated_triton_func = softmax_triton
generated_torch_func = softmax_torch

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048];
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
    benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes]);
