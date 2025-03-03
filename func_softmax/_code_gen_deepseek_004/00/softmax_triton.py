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
    Computes softmax for a 2D tensor row-wise using Triton.

    Args:
        output_ptr (tl.tensor): Output tensor pointer [batch_size, dim_size]
        input_ptr (tl.tensor): Input tensor pointer [batch_size, dim_size]
        input_row_stride (int): Input tensor row stride
        output_row_stride (int): Output tensor row stride
        n_cols (int): Number of columns in each row
        BLOCK_SIZE (tl.constexpr): Processing block size (must be power of two)

    Returns:
        None: Output is stored directly in memory

    Explanation:
        1. Each program processes one row of the input tensor
        2. First pass computes maximum value in chunks of BLOCK_SIZE
        3. Second pass calculates sum of exponentials in chunks
        4. Final pass computes and stores normalized softmax values
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Find row maximum
    max_val = -tl.inf(tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=-tl.inf(tl.float32))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Compute exponentials and their sum
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Normalize and store results
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=0)
        exp_vals = tl.exp(vals - max_val)
        softmax_vals = exp_vals / sum_exp
        tl.store(output_ptr + row_start + cols, softmax_vals, mask=mask)

def generated_triton_func(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of softmax function for 2D tensors.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output tensor [batch_size, dim_size]

    Explanation:
        1. Determines optimal block size based on column dimension
        2. Configures kernel launch parameters (grid, block size)
        3. Allocates output tensor and launches kernel
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4
    
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return y

def generated_torch_func(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of softmax.

    Args:
        x (torch.Tensor): Input tensor [batch_size, dim_size]

    Returns:
        torch.Tensor: Softmax output tensor [batch_size, dim_size]

    Explanation:
        1. Directly uses optimized PyTorch softmax implementation
        2. Applies along last dimension (-1)
    """
    return torch.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, dim_size, device=device, dtype=dtype)
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    verify_correctness_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(generated_triton_func, generated_torch_func, input_generator, [batch_sizes, dim_sizes])
