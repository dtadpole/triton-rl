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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes softmax probabilities for each row in a 2D tensor.

    Args:
        output_ptr (triton.PointerType): Output tensor pointer. Shape (batch_size, dim_size)
        input_ptr (triton.PointerType): Input tensor pointer. Shape (batch_size, dim_size)
        input_row_stride (int): Input tensor row stride in elements
        output_row_stride (int): Output tensor row stride in elements
        n_cols (int): Number of columns/dimension size
        BLOCK_SIZE (tl.constexpr): Processing block size (triton constant)

    Returns:
        None: Results stored directly in output_ptr

    Explanation:
        1. Each program processes one row using program_id(0)
        2. Three-phase computation:
           a. Find row maximum for numerical stability
           b. Compute sum of exponentials after subtracting max
           c. Calculate final softmax values by dividing exp(x) by sum
        3. Uses block-wise processing for memory efficiency
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Phase 1: Compute row max
    max_val = -float('inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Phase 2: Compute exponential sum
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Phase 3: Compute final softmax
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        vals = tl.load(input_ptr + row_start + cols, mask=mask)
        exp_vals = tl.exp(vals - max_val)
        softmax_out = exp_vals / sum_exp
        tl.store(output_ptr + row_idx * output_row_stride + cols, softmax_out, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated softmax implementation for 2D tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax probabilities with same shape as input

    Explanation:
        1. Auto-tunes block size based on column dimension
        2. Launches parallel kernels for each row
        3. Uses Torch tensor memory layout directly
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = max(min(BLOCK_SIZE, 1024), 128)  # Tuned performance constraints
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](y, x, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Reference PyTorch softmax implementation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax probabilities with same shape as input
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        return torch.randn(batch_size, dim_size, device=device, dtype=dtype)

    from benchmark import verify_correctness_func, benchmark_performance_func

    verify_correctness_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(triton_softmax, torch_softmax, input_generator, [batch_sizes, dim_sizes])
