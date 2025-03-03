# filename: softmax_triton.py
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Computes softmax along the last dimension of a 2D tensor using Triton.

    Args:
        x_ptr (tl.tensor): Input tensor pointer. Shape (n_rows, n_cols)
        y_ptr (tl.tensor): Output tensor pointer. Shape (n_rows, n_cols)
        n_cols (int): Number of columns in the tensor
        BLOCK_SIZE (tl.constexpr): Block size for column processing

    Returns:
        None: Output is written directly to y_ptr

    Explanation:
        1. Determine current row index using program ID
        2. Generate column offsets and validity mask
        3. Load input row data with boundary check
        4. Compute row maximum for numerical stability
        5. Subtract max and compute exponentials
        6. Calculate sum of exponentials
        7. Normalize each element by sum
        8. Store results to output
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_ptrs = x_ptr + row_idx * n_cols + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(x, axis=0)
    x_minus_max = x - row_max
    exp_x = tl.exp(x_minus_max)

    row_sum = tl.sum(exp_x, axis=0)
    y = exp_x / row_sum

    y_ptrs = y_ptr + row_idx * n_cols + col_offsets
    tl.store(y_ptrs, y, mask=mask)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax along last dimension using Triton optimized kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied, same shape as input

    Explanation:
        1. Extract tensor dimensions
        2. Calculate optimal block size (nearest power of two)
        3. Determine number of warps based on block size
        4. Allocate output tensor
        5. Configure and launch Triton kernel
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 8

    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        x, y, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Applies standard PyTorch softmax along last dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Output tensor with softmax applied, same shape as input

    Explanation:
        1. Directly uses torch.nn.functional.softmax with dim=-1
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    from benchmark import verify_correctness, benchmark_performance

    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x

    verify_correctness(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
