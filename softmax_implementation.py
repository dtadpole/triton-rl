# filename: softmax_implementation.py
import torch
import triton
import triton.language as tl

# PyTorch reference implementation
def torch_softmax(x):
    """Compute softmax using PyTorch's native implementation."""
    return torch.nn.functional.softmax(x, dim=-1)

# Triton implementation of softmax
@triton.jit
def _softmax_kernel(
    output_ptr, input_ptr,
    batch_size, dim_size,
    stride_batch, stride_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)

    # Check if we're in bounds
    if pid >= batch_size:
        return

    # Compute the row offset for this program ID
    row_start = pid * stride_batch

    # Create offsets for this row
    offsets = row_start + tl.arange(0, BLOCK_SIZE) * stride_dim

    # Create a mask for valid elements in this row
    mask = tl.arange(0, BLOCK_SIZE) < dim_size

    # Load input data with mask
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))

    # Compute row maximum for numerical stability
    row_max = tl.max(x, axis=0)

    # Subtract maximum from each element for numerical stability
    x = x - row_max

    # Compute exponentials
    x = tl.exp(x)

    # Compute sum of exponentials
    row_sum = tl.sum(x, axis=0)

    # Normalize by the sum
    x = x / row_sum

    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)

def triton_softmax(x):
    """Compute softmax using Triton implementation."""
    batch_size, dim_size = x.shape

    # Allocate output tensor
    output = torch.empty_like(x)

    # Calculate strides
    stride_batch = x.stride(0)
    stride_dim = x.stride(1) if dim_size > 1 else 1

    # Determine block size based on dimension size (power of 2 for efficiency)
    BLOCK_SIZE = triton.next_power_of_2(dim_size)

    # Launch kernel
    grid = (batch_size,)
    _softmax_kernel[grid](
        output, x,
        batch_size, dim_size,
        stride_batch, stride_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

# Export for benchmarking
generated_triton_func = triton_softmax
generated_torch_func = torch_softmax
