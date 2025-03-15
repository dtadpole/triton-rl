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
    Computes row-wise softmax using three-step Triton kernel.

    Args:
        output_ptr (triton.Pointer): Output tensor pointer. Shape (n_rows, n_cols)
        input_ptr (triton.Pointer): Input tensor pointer. Shape (n_rows, n_cols)
        input_row_stride (int): Stride between input rows (elements)
        output_row_stride (int): Stride between output rows (elements)
        n_cols (int): Number of columns per row (dim_size)
        BLOCK_SIZE (tl.constexpr): Processing block size (triton constant)

    Returns:
        None: Output is written directly to memory

    Explanation:
        1. Iterate through row in BLOCK_SIZE chunks to find row maximum
        2. Second pass calculates sum of exponentials (max stabilized)
        3. Final pass computes and stores normalized softmax values
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride

    # Step 1: Find row maximum
    max_val = -tl.infinity(tl.float32)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        ptrs = input_ptr + row_start + cols
        vals = tl.load(ptrs, mask=mask, other=-tl.infinity(tl.float32))
        max_val = tl.maximum(max_val, tl.max(vals, 0))

    # Step 2: Compute sum of exponentials
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        ptrs = input_ptr + row_start + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, 0)

    # Step 3: Compute and store softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        ptrs = input_ptr + row_start + cols
        vals = tl.load(ptrs, mask=mask, other=0.0)
        softmax_vals = tl.exp(vals - max_val) / sum_exp
        
        out_ptrs = output_ptr + row_idx * output_row_stride + cols
        tl.store(out_ptrs, softmax_vals, mask=mask)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of row-wise softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)

    Returns:
        torch.Tensor: Softmax-normalized tensor of same shape as input

    Explanation:
        1. Allocates output tensor matching input shape
        2. Configures kernel launch parameters (grid, block size)
        3. Invokes Triton kernel with input/output tensor metadata
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    y = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def softmax_torch(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of row-wise softmax.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim_size)
    
    Returns:
        torch.Tensor: Softmax-normalized tensor of same shape as input
    
    Explanation:
        1. Directly applies PyTorch's softmax along last dimension
    """
    return torch.nn.functional.softmax(x, dim=-1)

if __name__ == "__main__":
    batch_sizes = [256, 512, 1024, 2048]
    dim_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    
    def input_generator(batch_size, dim_size, device='cuda', dtype=torch.float16):
        x = torch.randn(batch_size, dim_size, device=device, dtype=dtype)
        return x
    
    from benchmark import verify_correctness_func, benchmark_performance_func
    
    verify_correctness_func(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
    benchmark_performance_func(softmax_triton, softmax_torch, input_generator, [batch_sizes, dim_sizes])
