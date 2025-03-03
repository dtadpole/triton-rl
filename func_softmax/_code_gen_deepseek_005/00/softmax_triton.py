# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes row-wise softmax activation for a 2D tensor using Triton.

    Args:
        output_ptr (triton.PointerType): Output tensor pointer. Shape (batch, n_cols)
        input_ptr (triton.PointerType): Input tensor pointer. Shape (batch, n_cols)
        input_row_stride (int): Input tensor row stride in elements
        output_row_stride (int): Output tensor row stride in elements
        n_cols (int): Number of columns per row
        BLOCK_SIZE (tl.constexpr): Processing block size (elements per iteration)

    Returns:
        None: Output is stored through output_ptr

    Explanation:
        1. Calculate row index from program ID
        2. Find maximum value in row through block-reduce
        3. Compute sum of exp(x_i - max) for normalization
        4. Store normalized exp(x_i - max)/sum to output
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    output_row_start = row_idx * output_row_stride

    # Find row maximum
    max_val = float('-inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Compute normalization sum
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Write final normalized values
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0)
        exp_vals = tl.exp(vals - max_val)
        softmax_out = exp_vals / sum_exp
        tl.store(output_ptr + output_row_start + cols, softmax_out, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied row-wise. Shape matches input.

    Explanation:
        1. Validate input dimensions
        2. Configure kernel launch parameters
        3. Execute Triton kernel with automatic memory management
    """
    batch_size, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    y = torch.empty_like(x)
    grid = (batch_size,)
    
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax using PyTorch.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied row-wise. Shape matches input.

    Explanation:
        1. Uses PyTorch's built-in softmax implementation
    """
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
