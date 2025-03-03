# filename: softmax_triton.py
import torch
import triton
import triton.language as tl
from benchmark import verify_correctness, benchmark_performance

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Applies the softmax function to each row of the input tensor.

    Args:
        output_ptr (triton.Pointer): Output tensor pointer. Shape (n_rows, n_cols)
        input_ptr (triton.Pointer): Input tensor pointer. Shape (n_rows, n_cols)
        n_cols (int): Number of columns in each row
        BLOCK_SIZE (tl.constexpr): Processing block size for element chunks

    Returns:
        None: Output is stored through output_ptr

    Explanation:
        1. Compute row index and starting position
        2. Find maximum value via block-wise reduction
        3. Calculate sum of exponentials (x - max)
        4. Normalize and store results
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    # Compute max value
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        values = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
        curr_max = tl.max(values, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # Compute sum of exponentials
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        values = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        exp_values = tl.exp(values - max_val)
        sum_exp += tl.sum(exp_values, axis=0)

    # Compute normalized values
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start + cols
        output_ptrs = output_ptr + row_start + cols
        values = tl.load(input_ptrs, mask=mask).to(tl.float32)
        exp_values = tl.exp(values - max_val)
        normalized = exp_values / sum_exp
        tl.store(output_ptrs, normalized, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Computes softmax along last dimension using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax output with same shape as input

    Explanation:
        1. Initialize output tensor
        2. Determine input dimensions
        3. Configure block size (power-of-two up to 4096)
        4. Launch kernel with row-parallel execution grid
    """
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4096)
    grid = (n_rows,)
    softmax_kernel[grid](output, x, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output

batch_sizes = [256, 512, 1024, 2048]
dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
    return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

def generated_torch_func(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    verify_correctness(triton_softmax, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(triton_softmax, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
