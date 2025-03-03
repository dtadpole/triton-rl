# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes softmax for each row of the input tensor using Triton.

    Args:
        output_ptr (triton.PointerType): Pointer to the output tensor. Shape (batch_size, dim_size)
        input_ptr (triton.PointerType): Pointer to the input tensor. Shape (batch_size, dim_size)
        n_cols (int): Number of columns (dim_size) in each row
        input_row_stride (int): Stride between rows in input tensor
        output_row_stride (int): Stride between rows in output tensor
        BLOCK_SIZE (tl.constexpr): Block size for kernel execution

    Returns:
        None: Output is written directly to output_ptr

    Explanation:
        1. Determine row index based on program ID
        2. Iterate over row elements in blocks to compute max value
        3. Compute sum of exponentials (value - max)
        4. Normalize each element by sum and store result
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_offsets = row_start + col_offsets

    # Find max value in row
    max_val = tl.float32(-float('inf'))
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        val = tl.load(input_ptr + cols, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(val, 0))

    # Compute sum of exponentials
    sum_exp = tl.float32(0)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        val = tl.load(input_ptr + cols, mask=mask, other=0)
        exp_val = tl.exp(val - max_val)
        sum_exp += tl.sum(exp_val, 0)

    # Normalize and store results
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + col_offsets
        mask = cols < n_cols
        val = tl.load(input_ptr + cols, mask=mask, other=0)
        exp_val = tl.exp(val - max_val)
        softmax_val = exp_val / sum_exp
        tl.store(output_ptr + cols, softmax_val, mask=mask)

def generated_triton_func(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax to input tensor using Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor after softmax, same shape as input

    Explanation:
        1. Determine tensor dimensions and optimal block size
        2. Allocate output tensor
        3. Configure and launch Triton kernel
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        n_cols,
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def generated_torch_func(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax using PyTorch's native implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor after softmax, same shape as input

    Explanation:
        1. Use PyTorch's built-in softmax with dim=-1
    """
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
